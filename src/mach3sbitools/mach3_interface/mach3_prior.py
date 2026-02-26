"""
Helper module to create SBI-compatible priors from MaCh3 parameter information
with parameter scaling for improved neural network training using PyTorch transforms.
"""

import math
import torch
import fnmatch
from typing import List, Tuple, Optional

from torch.distributions import Independent, Normal, Uniform, constraints


class MaCh3Prior(torch.distributions.Distribution):
    """
    Creates an SBI-compatible prior from MaCh3 parameter information.

    Handles mixed priors where some parameters have Gaussian priors (with errors)
    and others have flat/uniform priors (indicated by error = -1).

    Supports parameter scaling to normalise parameters to similar scales for better
    neural network training in SBI using PyTorch's built-in TransformedDistribution.
    """

    def __init__(
        self,
        nominals: List[float],
        errors: List[float],
        bounds: Tuple[List[float], List[float]],
        flat_pars: List[bool],
        nuisance_pars: Optional[List[str]],
        cyclical_pars: Optional[List[str]] = None,
        parameter_names: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            nominals: List of nominal/central values for each parameter
            errors: List of errors (1-sigma) for Gaussian priors, or -1 for flat priors
            bounds: Tuple of (lower_bounds, upper_bounds) for each parameter
            flat_pars: List of booleans indicating flat (True) vs Gaussian (False) priors
            nuisance_pars: Optional list of parameter name patterns (fnmatch) to exclude
            cyclical_pars: Optional list of parameter name patterns (fnmatch) that should
                receive a cyclical sinusoidal prior.  Their bounds are overridden to
                (−2π, +2π) regardless of what is supplied in ``bounds``.  Each matched
                parameter must be flat; a ValueError is raised if it is not.
            parameter_names: Optional list of parameter names for reference
            device: Device to place tensors on ('cpu', 'cuda', 'mps')
        """
        self.parameter_names = parameter_names or [f"param_{i}" for i in range(len(nominals))]

        if nuisance_pars is not None:
            mask = [not any(fnmatch.fnmatch(name, n) for n in nuisance_pars) for name in self.parameter_names]
        else:
            mask = [True] * len(self.parameter_names)

        # Override bounds to ±2π for any parameter matching a cyclical_pars pattern
        lower = list(bounds[0])
        upper = list(bounds[1])
        if cyclical_pars is not None:
            for i, name in enumerate(self.parameter_names):
                if any(fnmatch.fnmatch(name, pattern) for pattern in cyclical_pars):
                    if not flat_pars[i]:
                        raise ValueError(
                            f"Parameter '{name}' matched cyclical_pars but is not flat. "
                            "Only flat parameters can use the cyclical prior."
                        )
                    lower[i] = -2.0 * math.pi
                    upper[i] =  2.0 * math.pi

        self.device = device
        self.nominals     = torch.tensor(nominals, dtype=torch.float32, device=device)[mask]
        self.errors       = torch.tensor(errors,   dtype=torch.float32, device=device)[mask]
        self.lower_bounds = torch.tensor(lower,    dtype=torch.float32, device=device)[mask]
        self.upper_bounds = torch.tensor(upper,    dtype=torch.float32, device=device)[mask]

        self.n_params = len(nominals)

        # Identify which parameters have Gaussian vs flat priors
        self.flat_mask     = torch.tensor(flat_pars, dtype=torch.bool)[mask]
        self.gaussian_mask = ~self.flat_mask

        self.n_gaussian = self.gaussian_mask.sum().item()
        self.n_flat     = self.flat_mask.sum().item()

        if self.n_gaussian + self.n_flat != self.n_params:
            raise ValueError("Each parameter must be either Gaussian (error > 0) or flat (error = -1)")

        # Build the base prior distribution (in original space)
        self._build_prior()

        # Initialise parent class
        super().__init__(
            batch_shape=self._prior.batch_shape,
            event_shape=self._prior.event_shape,
            validate_args=False,
        )

    def _build_prior(self):
        """Construct the base prior distribution in original parameter space."""

        if self.n_flat == 0:
            # All Gaussian
            self._prior = BoundedGaussianPrior(
                nominals=self.nominals,
                errors=self.errors,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                device=self.device,
            )
            self.prior_type = "gaussian"

        elif self.n_gaussian == 0:
            # All flat: must still go through MixedPrior so that any cyclical
            # parameters (detected by their ±2π bounds) get CyclicalPrior rather
            # than a plain Uniform.
            self._prior = MixedPrior(
                nominals=self.nominals,
                errors=self.errors,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                gaussian_mask=self.gaussian_mask,
                flat_mask=self.flat_mask,
                device=self.device,
            )
            self.prior_type = "uniform"

        else:
            # Mixed Gaussian + flat
            self._prior = MixedPrior(
                nominals=self.nominals,
                errors=self.errors,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                gaussian_mask=self.gaussian_mask,
                flat_mask=self.flat_mask,
                device=self.device,
            )
            self.prior_type = "mixed"

    @property
    def support(self):
        return self._prior.support

    def rsample(self, sample_shape=torch.Size()):
        """Sample with reparameterization (required by some SBI methods)."""
        return self._prior.rsample(sample_shape)

    def sample(self, sample_shape=torch.Size([])):
        return self._prior.sample(sample_shape)

    def log_prob(self, value):
        return self._prior.log_prob(value)

    def check_bounds(self, params: torch.Tensor) -> torch.Tensor:
        """
        Check if parameters are within bounds.

        Args:
            params: Parameters to check (shape: [..., n_params])

        Returns:
            Boolean tensor indicating which samples are in bounds (shape: [...])
        """
        in_bounds = (params >= self.lower_bounds) & (params <= self.upper_bounds)
        return in_bounds.all(dim=-1)


# ---------------------------------------------------------------------------
# Component distributions
# ---------------------------------------------------------------------------

class BoundedGaussianPrior(torch.distributions.Distribution):
    """
    Independent Gaussian distributions with hard bounds.

    Samples outside bounds have -inf log probability.
    Sampling uses rejection sampling to ensure all samples are within bounds.
    """

    def __init__(
        self,
        nominals: torch.Tensor,
        errors: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        device: str = "cpu",
    ):
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([len(nominals)]),
            validate_args=False,
        )
        self.device       = device
        self.nominals     = nominals.to(device)
        self.errors       = errors.to(device)
        self.lower_bounds = lower_bounds.to(device)
        self.upper_bounds = upper_bounds.to(device)
        self.base_dist    = Independent(Normal(loc=nominals, scale=errors), 1)

    @property
    def support(self):
        return constraints.independent(
            constraints.interval(self.lower_bounds, self.upper_bounds), 1
        )

    def sample(self, sample_shape=torch.Size([])):
        """Sample using rejection sampling."""
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(
                [sample_shape] if isinstance(sample_shape, int) else sample_shape
            )

        samples   = torch.zeros(sample_shape + self.event_shape, device=self.device)
        remaining = torch.ones(sample_shape, dtype=torch.bool, device=self.device)

        for _ in range(10_000):
            if not remaining.any():
                break
            n_remaining = remaining.sum().item()
            new_samples = self.base_dist.sample(torch.Size([n_remaining]))
            in_bounds   = (
                (new_samples >= self.lower_bounds) & (new_samples <= self.upper_bounds)
            ).all(dim=-1)

            remaining_indices = torch.where(remaining.flatten())[0]
            valid_indices     = remaining_indices[in_bounds]

            if len(sample_shape) == 0:
                if in_bounds.any():
                    samples   = new_samples[in_bounds][0]
                    remaining = torch.tensor(False, device=self.device)
            else:
                samples.view(-1, samples.shape[-1])[valid_indices] = new_samples[in_bounds]
                remaining.view(-1)[valid_indices] = False

        if remaining.any():
            raise RuntimeError(
                "Rejection sampling failed after 10,000 iterations. "
                "Bounds may be too tight relative to the Gaussian width."
            )
        return samples

    def rsample(self, sample_shape=torch.Size([])):
        return self.sample(sample_shape)

    def log_prob(self, value):
        if value.device.type != self.device:
            value = value.to(self.device)
        log_p     = self.base_dist.log_prob(value)
        in_bounds = (
            (value >= self.lower_bounds) & (value <= self.upper_bounds)
        ).all(dim=-1)
        return torch.where(in_bounds, log_p, torch.tensor(float("-inf"), device=self.device))


class CyclicalPrior(torch.distributions.Distribution):
    """
    Cyclical prior for a *single* scalar parameter over the range (−2π, 2π).

    The unnormalised density is:

        f(θ) = ½ · sin²( (θ + 2π) / 4 )

    Integrating over (−2π, 2π) gives π, so the correctly normalised PDF is:

        p(θ) = 1/(2π) · sin²( (θ + 2π) / 4 ),   θ ∈ (−2π, 2π)

    Properties
    ----------
    - Zero at both endpoints θ = ±2π.
    - Symmetric about θ = 0, with maximum p(0) = 1/(2π).
    - Mean = 0 (by symmetry).
    - Variance = 4π² − 8 ≈ 31.48 (computed analytically).

    Sampling
    --------
    Uses exact inverse-CDF sampling via vectorised bisection (60 iterations,
    absolute accuracy < 4π / 2^60 ≈ 1 × 10⁻¹⁷). No rejection sampling needed.
    """

    _LOW  = -2.0 * math.pi
    _HIGH =  2.0 * math.pi

    def __init__(self, device: str = "cpu"):
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size(),
            validate_args=False,
        )
        self.device = device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cdf(theta: torch.Tensor) -> torch.Tensor:
        """
        Analytical CDF of the cyclical prior.

        Derivation
        ----------
        Let  u = (θ + 2π) / 4,  so dθ = 4 du.
        At the lower limit θ = −2π:  u_lo = 0.

            ∫ ½ sin²(u) · 4 du  =  2 [ u/2 − sin(2u)/4 ]  =  u − sin(2u)/2

        The primitive at u_lo = 0 is zero, so:

            CDF(θ) = [ u − sin(2u)/2 ] / π

        where dividing by π (the total integral) normalises to [0, 1].
        """
        u = (theta + 2.0 * math.pi) / 4.0
        # primitive at u_lo = 0 is 0, so no offset term needed
        return (u - torch.sin(2.0 * u) / 2.0) / math.pi

    @staticmethod
    def _inv_cdf(p: torch.Tensor, n_iter: int = 60) -> torch.Tensor:
        """
        Invert the CDF via vectorised bisection.

        60 iterations give absolute accuracy < 4π / 2^60 ≈ 1 × 10⁻¹⁷.
        """
        lo = torch.full_like(p, -2.0 * math.pi)
        hi = torch.full_like(p,  2.0 * math.pi)

        for _ in range(n_iter):
            mid      = (lo + hi) * 0.5
            go_right = CyclicalPrior._cdf(mid) < p
            lo       = torch.where(go_right, mid, lo)
            hi       = torch.where(go_right, hi,  mid)

        return (lo + hi) * 0.5

    # ------------------------------------------------------------------
    # torch.distributions interface
    # ------------------------------------------------------------------

    @property
    def support(self) -> constraints.Constraint:
        return constraints.interval(self._LOW, self._HIGH)

    @property
    def mean(self) -> torch.Tensor:
        """Mean = 0 by symmetry."""
        return torch.tensor(0.0, device=self.device)

    @property
    def variance(self) -> torch.Tensor:
        """
        Variance computed analytically:

            ∫_{-2π}^{2π} θ² · p(θ) dθ  =  4π² − 8  ≈  31.48
        """
        return torch.tensor(4.0 * math.pi ** 2 - 8.0, device=self.device)

    def sample(self, sample_shape=torch.Size([])) -> torch.Tensor:
        """Draw samples via exact inverse-CDF (no rejection needed)."""
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(
                [sample_shape] if isinstance(sample_shape, int) else sample_shape
            )
        with torch.no_grad():
            u = torch.rand(sample_shape, device=self.device)
            return self._inv_cdf(u)

    def rsample(self, sample_shape=torch.Size([])) -> torch.Tensor:
        """
        Reparameterised sample.

        The inverse-CDF transform is differentiable everywhere inside the
        open support, so gradients flow through correctly.
        """
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(
                [sample_shape] if isinstance(sample_shape, int) else sample_shape
            )
        u = torch.rand(sample_shape, device=self.device)
        return self._inv_cdf(u)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Log probability under the cyclical prior.

            log p(θ) = −log(2π) + 2 · log|sin((θ + 2π)/4)|

        Returns -inf for θ outside (−2π, 2π) and at the endpoints where
        the density is zero.
        """
        if value.device.type != self.device:
            value = value.to(self.device)

        in_bounds = (value > self._LOW) & (value < self._HIGH)

        sin_val  = torch.sin((value + 2.0 * math.pi) / 4.0)
        # Substitute 1.0 outside bounds to avoid log(0) before the final mask
        safe_sin = torch.where(in_bounds, sin_val.abs(), torch.ones_like(sin_val))

        log_p = -math.log(2.0 * math.pi) + 2.0 * torch.log(safe_sin)
        return torch.where(in_bounds, log_p, torch.tensor(float("-inf"), device=self.device))


class MixedPrior(torch.distributions.Distribution):
    """
    Custom distribution for mixed Gaussian, flat/uniform, and cyclical priors.

    Each parameter is assigned one of three prior types:

    - **Gaussian** (``flat_mask = False``, ``error > 0``): bounded Normal,
      sampled via rejection sampling within the parameter bounds.
    - **Cyclical** (``flat_mask = True``, bounds ≈ (−2π, +2π)): the
      sinusoidal prior  p(θ) ∝ sin²((θ + 2π)/4); sampled via exact inverse-CDF.
    - **Flat / Uniform** (``flat_mask = True``, all other bounds): Uniform
      over the parameter bounds, always in-bounds.

    Cyclical detection
    ------------------
    A flat parameter is treated as cyclical when its bounds match (−2π, +2π)
    within a tolerance of 1 × 10⁻⁴.  All other flat parameters use Uniform.
    """

    _CYCLICAL_TOL = 1e-4

    def __init__(
        self,
        nominals: torch.Tensor,
        errors: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        gaussian_mask: torch.Tensor,
        flat_mask: torch.Tensor,
        device: str = "cpu",
    ):
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([len(nominals)]),
            validate_args=False,
        )
        self.device        = device
        self.nominals      = nominals.to(device)
        self.errors        = errors.to(device)
        self.lower_bounds  = lower_bounds.to(device)
        self.upper_bounds  = upper_bounds.to(device)
        self.gaussian_mask = gaussian_mask.to(device)
        self.flat_mask     = flat_mask.to(device)

        self.gaussian_indices = torch.where(gaussian_mask)[0]
        self.flat_indices     = torch.where(flat_mask)[0]

        # Build per-parameter component distributions and record cyclical membership
        self.component_dists = []
        cyclical_flags = []

        for i in range(len(nominals)):
            if gaussian_mask[i]:
                self.component_dists.append(Normal(loc=nominals[i], scale=errors[i]))
                cyclical_flags.append(False)
            elif self._is_cyclical_bounds(lower_bounds[i], upper_bounds[i]):
                self.component_dists.append(CyclicalPrior(device=device))
                cyclical_flags.append(True)
            else:
                self.component_dists.append(Uniform(low=lower_bounds[i], high=upper_bounds[i]))
                cyclical_flags.append(False)

        self.cyclical_mask = torch.tensor(cyclical_flags, dtype=torch.bool, device=device)

    def _is_cyclical_bounds(self, lo: torch.Tensor, hi: torch.Tensor) -> bool:
        """Return True when the bounds match (−2π, +2π) within tolerance."""
        return (
            abs(lo.item() - (-2.0 * math.pi)) < self._CYCLICAL_TOL
            and abs(hi.item() - ( 2.0 * math.pi)) < self._CYCLICAL_TOL
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def support(self):
        return constraints.independent(
            constraints.interval(self.lower_bounds, self.upper_bounds), 1
        )

    @property
    def mean(self):
        means = torch.zeros_like(self.nominals)
        means[self.gaussian_mask] = self.nominals[self.gaussian_mask]
        flat_uniform = self.flat_mask & ~self.cyclical_mask
        if flat_uniform.any():
            means[flat_uniform] = (
                self.lower_bounds[flat_uniform] + self.upper_bounds[flat_uniform]
            ) / 2.0
        # Cyclical mean = 0, already initialised
        return means

    @property
    def variance(self):
        variances = torch.zeros_like(self.nominals)
        if self.gaussian_mask.any():
            variances[self.gaussian_mask] = self.errors[self.gaussian_mask] ** 2
        flat_uniform = self.flat_mask & ~self.cyclical_mask
        if flat_uniform.any():
            variances[flat_uniform] = (
                (self.upper_bounds[flat_uniform] - self.lower_bounds[flat_uniform]) ** 2 / 12.0
            )
        if self.cyclical_mask.any():
            variances[self.cyclical_mask] = 4.0 * math.pi ** 2 - 8.0
        return variances

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, sample_shape=torch.Size([])):
        """
        Sample from the mixed prior.

        - Uniform parameters  : direct sampling (always in bounds).
        - Cyclical parameters : exact inverse-CDF (no rejection needed).
        - Gaussian parameters : rejection sampling within bounds.
        """
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(
                [sample_shape] if isinstance(sample_shape, int) else sample_shape
            )

        n       = len(self.nominals)
        samples = torch.zeros(sample_shape + (n,), device=self.device)

        # Uniform (flat, non-cyclical) parameters
        for i in self.flat_indices:
            if not self.cyclical_mask[i]:
                samples[..., i] = self.component_dists[i].sample(sample_shape)

        # Cyclical parameters
        for i in torch.where(self.cyclical_mask)[0]:
            samples[..., i] = self.component_dists[i].sample(sample_shape)

        # Gaussian parameters with rejection sampling
        if len(self.gaussian_indices) > 0:
            n_gauss   = len(self.gaussian_indices)
            remaining = torch.ones(
                sample_shape + (n_gauss,), dtype=torch.bool, device=self.device
            )

            for _ in range(10_000):
                if not remaining.any():
                    break

                for idx, i in enumerate(self.gaussian_indices):
                    n_rem = remaining[..., idx].sum().item()
                    if n_rem == 0:
                        continue

                    new_s     = self.component_dists[i].sample(torch.Size([n_rem]))
                    in_bounds = (new_s >= self.lower_bounds[i]) & (new_s <= self.upper_bounds[i])

                    rem_idx   = torch.where(remaining[..., idx].flatten())[0]
                    valid_idx = rem_idx[in_bounds]

                    samples.view(-1, n)[valid_idx, i]           = new_s[in_bounds]
                    remaining.view(-1, n_gauss)[valid_idx, idx] = False

            if remaining.any():
                raise RuntimeError(
                    "Rejection sampling failed after 10,000 iterations. "
                    "Bounds may be too tight relative to the Gaussian width."
                )

        return samples

    def rsample(self, sample_shape=torch.Size([])):
        return self.sample(sample_shape)

    # ------------------------------------------------------------------
    # Log probability
    # ------------------------------------------------------------------

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute joint log probability across all parameters.

        Returns -inf if any parameter is outside its bounds.
        CyclicalPrior.log_prob already handles its own bound checking,
        so explicit checks are only needed for Gaussian and Uniform components.
        """
        if value.device.type != self.device:
            value = value.to(self.device)

        log_prob = torch.zeros(value.shape[:-1], device=self.device)

        for i, dist in enumerate(self.component_dists):
            param_val = value[..., i]
            lp        = dist.log_prob(param_val)

            if not self.cyclical_mask[i]:
                in_bounds = (
                    (param_val >= self.lower_bounds[i]) & (param_val <= self.upper_bounds[i])
                )
                lp = torch.where(
                    in_bounds, lp, torch.tensor(float("-inf"), device=self.device)
                )

            log_prob = log_prob + lp

        return log_prob


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_mach3_prior(
    mach3_wrapper,
    device: str = "cpu",
    nuisance_pars: List[str] | None = None,
    cyclical_pars: List[str] | None = None,
) -> MaCh3Prior:
    """
    Convenience function to create a prior from a MaCh3DUNEWrapper instance.

    Args:
        mach3_wrapper: Instance of MaCh3DUNEWrapper
        device: Device to place tensors on ('cpu', 'cuda', 'mps')
        nuisance_pars: Optional list of parameter name patterns (fnmatch) to exclude
        cyclical_pars: Optional list of parameter name patterns (fnmatch) that should
            receive a cyclical sinusoidal prior.  Forwarded directly to MaCh3Prior.

    Returns:
        MaCh3Prior object compatible with SBI
    """
    nominals, errors = mach3_wrapper.get_nominal_error()
    bounds           = mach3_wrapper.get_bounds()
    names            = mach3_wrapper.get_parameter_names()
    flat_pars        = [mach3_wrapper.get_is_flat(i) for i in range(len(names))]

    return MaCh3Prior(
        nominals=nominals,
        errors=errors,
        bounds=bounds,
        flat_pars=flat_pars,
        nuisance_pars=nuisance_pars,
        cyclical_pars=cyclical_pars,
        parameter_names=names,
        device=device,
    )