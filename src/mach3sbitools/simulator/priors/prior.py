"""
Main module for handling the simulator prior. Priors are of 3 types:
- Cyclical distribution. This uses the custom Cyclical Distribution defined in this module
- Flat (Uniform) distribution. This uses the torch Uniform distribution
- Gaussian distribution. This uses the torch MultivariateNormal distribution

Checks are done in the following order
- Cyclical: Check if anything is cyclical. Cyciclical parameters are forced to have bounds of ±2pi
- Flat: Check if anything is flat and NOT cyclical
- Gaussian: Everything else is assumed to be gaussian

"""

import fnmatch
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from torch.distributions import MultivariateNormal, Uniform, constraints

from mach3sbitools.utils import TorchDeviceHandler, get_logger

from ..simulator_injector import SimulatorProtocol
from .cyclical_distribution import CyclicalDistribution
from .dataclasses import PriorData

size_: TypeAlias = torch.Size | list[int] | tuple[int, ...]


class PriorNotFound(Exception): ...


logger = get_logger()


@dataclass(frozen=True)
class MaskDistributionMap:
    # Simple dataclass to store mask/distribution pairs
    mask: torch.Tensor
    distribution: torch.distributions.Distribution

    def to(self, device: torch.device) -> "MaskDistributionMap":
        return MaskDistributionMap(
            mask=self.mask.to(device), distribution=self.distribution
        )


class Prior(torch.distributions.Distribution):
    """
    The MaCh3SBITools prior. Designed to ~replicate prior construction in MaCh3 (https://github.com/mach3-software/MaCh3)
    Essentially a wrapper around cyclical, uniform and multivariate distributions.
    """

    nuisance_filter: torch.Tensor

    def __init__(
        self,
        prior_data: PriorData,
        flat_msk: list[bool] | None = None,
        cyclical_parameters: list[str] | None = None,
        nuisance_parameters: list[str] | None = None,
    ):
        """
        Prior constructor
        :param prior_data: The dataclass containing information about the prior
        :param flat_msk: A list of flat parameter indices. This requires EXACT name matches
        :param cyclical_parameters: A list of cyclical parameter names. This is checked with regex
        :param nuisance_parameters: A list of nuisance parameter names. This is checked with regex
        """
        self.device_handler = TorchDeviceHandler()

        # This means we can totally ignore nuisance params safely
        self._prior_data = prior_data

        # We can filter by nuisance params
        self.set_nuisance_filter(nuisance_parameters)

        # Select JUST cyclical parameters
        self._priors: list[MaskDistributionMap] = []

        cyclical_mask = torch.zeros(len(self.prior_data.nominals), dtype=torch.bool)
        if cyclical_parameters:
            cyclical_mask_ = [
                any(fnmatch.fnmatch(p, c) for c in cyclical_parameters)
                for p in self.prior_data.parameter_names
            ]
            cyclical_mask = self.device_handler.to_tensor(cyclical_mask_)

        if any(cyclical_mask):
            # Add cyclical params
            self._priors.append(self._get_cyclical_map(cyclical_mask))

        flat_mask = self.device_handler.to_tensor(flat_msk) & ~cyclical_mask
        # Make sure we don't confuse cyclical and flat params so we can do a bit more vectorisation

        if any(flat_mask):
            # Add flat params
            self._priors.append(self._get_flat_map(flat_mask))

        gaussian_mask = ~cyclical_mask & ~flat_mask
        if any(gaussian_mask):
            # Everything else is assumed gaussian
            self._priors.append(self._get_gaussian_map(gaussian_mask))

        # Finally we validate the superclass to stop us being complained at!
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([len(self.prior_data.nominals)]),
            validate_args=False,
        )

    # Parameter Setters
    def _get_cyclical_map(self, cyclical_mask):
        self.prior_data.lower_bounds[cyclical_mask] = -2 * torch.pi
        self.prior_data.upper_bounds[cyclical_mask] = 2 * torch.pi

        cyclical_data = self.prior_data[cyclical_mask]
        cyclical_dist = CyclicalDistribution(
            cyclical_data.nominals,
            cyclical_data.lower_bounds,
            cyclical_data.upper_bounds,
        )
        return MaskDistributionMap(cyclical_mask, cyclical_dist)

    def _get_flat_map(self, flat_mask):
        flat_data = self.prior_data[flat_mask]
        flat_dist = Uniform(flat_data.lower_bounds, flat_data.upper_bounds)
        return MaskDistributionMap(flat_mask, flat_dist)

    def _get_gaussian_map(self, gaussian_mask):
        gaussian_data = self.prior_data[gaussian_mask]
        gaussian_dist = MultivariateNormal(
            gaussian_data.nominals, covariance_matrix=gaussian_data.covariance_matrix
        )
        return MaskDistributionMap(gaussian_mask, gaussian_dist)

    @property
    def prior_data(self):
        return self._prior_data[self.nuisance_filter]

    def set_nuisance_filter(self, nuisance_patterns: list[str] | None = None) -> None:
        if nuisance_patterns is None:
            n_pars = len(self._prior_data.parameter_names)
            self.nuisance_filter = torch.ones(n_pars, dtype=torch.bool)
            return

        nuisance_filter_ = [
            not any(fnmatch.fnmatch(p, n) for n in nuisance_patterns)
            for p in self._prior_data.parameter_names  # ← iterate params, not patterns
        ]
        self.nuisance_filter = self.device_handler.to_tensor(nuisance_filter_)

    @property
    def mean(self) -> torch.Tensor:
        # Get the mean
        return self.device_handler.to_tensor(self.prior_data.nominals)

    @property
    def n_params(self) -> int:
        return len(self.prior_data.nominals)

    @property
    def variance(self) -> torch.Tensor:
        # Get the variance
        variance = torch.zeros(len(self.prior_data.nominals))
        for mask_map in self._priors:
            variance[mask_map.mask] = mask_map.distribution.variance
        return variance

    @property
    def support(self):
        return constraints.independent(
            constraints.interval(
                self.prior_data.lower_bounds, self.prior_data.upper_bounds
            ),
            1,
        )

    def sample(self, sample_shape=torch.Size([])):
        sample_shape = torch.Size(sample_shape)  # ← normalize tuple → torch.Size
        samples = torch.empty(*sample_shape, self.n_params, dtype=torch.double)
        for mask_map in self._priors:
            samples[..., mask_map.mask] = mask_map.distribution.sample(sample_shape).to(
                torch.double
            )
        return samples

    def rsample(self, sample_shape=torch.Size([])):
        sample_shape = torch.Size(sample_shape)
        samples = torch.empty(*sample_shape, self.n_params, dtype=torch.double)
        for mask_map in self._priors:
            samples[..., mask_map.mask] = mask_map.distribution.rsample(
                sample_shape
            ).to(torch.double)
        return samples

    def check_bounds(self, params: torch.Tensor) -> torch.Tensor:
        in_bounds = (params >= self.prior_data.lower_bounds) & (
            params <= self.prior_data.upper_bounds
        )
        return self.device_handler.to_tensor(in_bounds.all(dim=-1))

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(self, f)

    # Does to in place
    def to(self, device: torch.device) -> "Prior":
        self._prior_data = self._prior_data.to(device)
        for i, mask_map in enumerate(self._priors):
            self._priors[i] = mask_map.to(device)
        self.nuisance_filter = self.nuisance_filter.to(device)
        return self


def _check_boundary(
    nominal: torch.Tensor,
    error: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    parameter_names: np.ndarray,
) -> None:
    """
    print a warning if any parameters have massive boundaries
    """

    warning_thresh = 10

    warning_ub = nominal + error * warning_thresh
    warning_lb = nominal - error * warning_thresh

    mask = (lower_bound < warning_lb) | (upper_bound > warning_ub)
    if not any(mask):
        return

    logger.warning(
        f"The following parameters have boundaries > {warning_thresh:d}σ from their prior nominal"
    )
    parameter_names_masked = parameter_names[mask.cpu().numpy()]
    nominals_masked = nominal[mask]
    errors_masked = error[mask]
    lb_masked = lower_bound[mask]
    ub_masked = upper_bound[mask]

    for param_info in zip(
        parameter_names_masked, nominals_masked, errors_masked, lb_masked, ub_masked
    ):
        logger.warning(
            "   '{:s}' | Nominal: {:4f}, Error {:4f} | Lower Bnd {:4f}, Upper Bnd {:4f}".format(
                *param_info
            )
        )


def create_prior(
    simulator_instance: SimulatorProtocol,
    nuisance_pars: list[str] | None = None,
    cyclical_pars: list[str] | None = None,
) -> Prior:
    """
    Convenience function to create a prior from a MaCh3DUNEWrapper instance.

    Args:
        simulator_instance: Instance of MaCh3DUNEWrapper
        device: Device to place tensors on ('cpu', 'cuda', 'mps')
        nuisance_pars: Optional list of parameter name patterns (fnmatch) to exclude
        cyclical_pars: Optional list of parameter name patterns (fnmatch) that should
            receive a cyclical sinusoidal prior.  Forwarded directly to MaCh3Prior.

    Returns:
        MaCh3Prior object compatible with SBI
    """
    logger.info("Creating Prior")
    dh = TorchDeviceHandler()

    # Make everything a tensor
    nominals = dh.to_tensor(simulator_instance.get_parameter_nominals())
    errors = dh.to_tensor(simulator_instance.get_parameter_errors())
    lower_arr, upper_arr = simulator_instance.get_parameter_bounds()

    lower = dh.to_tensor(lower_arr)
    upper = dh.to_tensor(upper_arr)
    names = np.array(simulator_instance.get_parameter_names(), dtype=str)

    _check_boundary(nominals, errors, lower, upper, names)

    covariance = dh.to_tensor(simulator_instance.get_covariance_matrix())

    flat_pars = [simulator_instance.get_is_flat(i) for i in range(len(names))]

    data = PriorData(
        parameter_names=names,
        nominals=nominals,
        lower_bounds=lower,
        upper_bounds=upper,
        covariance_matrix=covariance,
    )

    return Prior(
        prior_data=data,
        flat_msk=flat_pars,
        nuisance_parameters=nuisance_pars,
        cyclical_parameters=cyclical_pars,
    )


def load_prior(prior_path: Path, device=torch.device("cpu")) -> Prior:
    """
    Load the prior from the given path
    """
    if not prior_path.is_file() or not prior_path.exists():
        raise PriorNotFound("Could not find prior %s", prior_path)

    with prior_path.open("rb") as f:
        prior = pickle.load(f)

    if not isinstance(prior, Prior):
        raise PriorNotFound(
            "No valid prior in %s. Instead found %s", prior_path, type(prior)
        )

    return prior.to(device)
