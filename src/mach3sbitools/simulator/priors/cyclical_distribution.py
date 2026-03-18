"""
Cyclical sinusoidal prior distribution.

Implements a custom :class:`torch.distributions.Distribution` for parameters
that wrap around periodically (e.g. angles), using a sinusoidal PDF over
``[-2π, 2π]``.
"""

import numpy as np
import torch
import torch.distributions


class CyclicalDistribution(torch.distributions.Distribution):
    """
    Sinusoidal prior for cyclical (periodic) parameters over ``[-2π, 2π]``.

    The probability density function is:

    .. math::

        p(\\theta) = \\frac{1}{2\\pi} \\sin^2\\!\\left(\\frac{\\theta + 2\\pi}{4}\\right),
        \\quad \\theta \\in [-2\\pi,\\, 2\\pi]

    Sampling uses inverse transform sampling via a precomputed CDF lookup table.

    .. note::

        Only bounds of exactly ``[-2π, 2π]`` are supported. Passing any other
        bounds raises :exc:`NotImplementedError`.
    """

    def __init__(
        self,
        nominals: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ):
        """
        Construct a :class:`CyclicalDistribution`.

        :param nominals: Nominal (centre) values, one per parameter.
        :param lower_bounds: Lower bounds — must all equal ``-2π``.
        :param upper_bounds: Upper bounds — must all equal ``+2π``.
        :raises NotImplementedError: If any bound differs from ``±2π``.
        """
        self.nominals = nominals
        self.device = nominals.device
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        if any(upper_bounds != 2 * torch.pi) or any(lower_bounds != -2 * torch.pi):
            raise NotImplementedError(
                "CyclicalDistribution only supports bounds of [-2π, 2π]."
            )

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([len(nominals)]),
            validate_args=False,
        )

    @property
    def mean(self):
        """Mean of the distribution (zero by symmetry)."""
        return 0

    @property
    def variance(self) -> torch.Tensor:
        """
        Variance of the distribution.

        Computed analytically as :math:`\\int p(x)\\,x^2\\,dx` over ``[-2π, 2π]``.
        """
        return torch.Tensor([5.16947])

    def pdf(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the probability density function at *theta*.

        :param theta: Input tensor. Values outside ``[-2π, 2π]`` have zero density.
        :returns: PDF values, same shape as *theta*.
        """
        in_bounds = (theta > self.lower_bounds) & (theta < self.upper_bounds)
        pdf = torch.zeros(theta.shape, dtype=torch.double)
        pdf[in_bounds] = (0.5 / torch.pi) * (
            torch.sin((theta[in_bounds] + 2 * torch.pi) / 4) ** 2
        )
        return pdf

    def cdf(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the cumulative distribution function at *theta*.

        :param theta: Input tensor.
        :returns: CDF values in ``[0, 1]``, same shape as *theta*.
        """
        cdf = torch.zeros(theta.shape, dtype=torch.double)
        in_bounds = (theta > self.lower_bounds) & (theta < self.upper_bounds)
        cdf[in_bounds] = (0.5 / torch.pi) * (
            theta[in_bounds] / 2 + torch.sin(theta[in_bounds] / 2) + torch.pi
        )
        cdf[theta > self.upper_bounds] = 1.0
        return cdf

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability density at *value*.

        Returns ``-inf`` for values outside ``[-2π, 2π]``.

        :param value: Input tensor.
        :returns: Log-density values, same shape as *value*.
        """
        pdf = self.pdf(value)
        in_bounds = torch.abs(pdf) > 1e-8
        pdf[in_bounds] = torch.log(pdf[in_bounds])
        pdf[~in_bounds] = -np.inf
        return pdf

    def _build_cdf_grid(
        self, n_points: int = 10_000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute a lookup table of ``(theta, CDF(theta))`` over ``[-2π, 2π]``.

        :param n_points: Number of grid points.
        :returns: Tuple of ``(theta_grid, cdf_grid)`` tensors.
        """
        theta_grid = torch.linspace(
            -2 * torch.pi,
            2 * torch.pi,
            n_points,
            dtype=torch.double,
            device=self.device,
        )
        cdf_grid = self.cdf(theta_grid)
        return theta_grid, cdf_grid

    def _sample_uniform_cdf(
        self, n_samples: int, cdf_min: torch.Tensor, cdf_max: torch.Tensor
    ) -> torch.Tensor:
        """
        Draw uniform samples in ``[cdf_min, cdf_max]`` for use in inverse CDF sampling.

        :param n_samples: Number of samples.
        :param cdf_min: Lower CDF bound (scalar or broadcastable tensor).
        :param cdf_max: Upper CDF bound (scalar or broadcastable tensor).
        :returns: Tensor of shape ``(n_samples, event_size)``.
        """
        event_size = len(self.nominals)
        u = torch.rand(n_samples, event_size, dtype=torch.double, device=self.device)
        return u * (cdf_max - cdf_min) + cdf_min

    def _invert_cdf(
        self, u: torch.Tensor, theta_grid: torch.Tensor, cdf_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Map uniform CDF values back to *theta* via the precomputed lookup table.

        Uses :func:`torch.searchsorted` on *cdf_grid*.

        :param u: Uniform samples in ``[0, 1]``.
        :param theta_grid: Grid of theta values.
        :param cdf_grid: Corresponding CDF values.
        :returns: Tensor of theta samples, same shape as *u*.
        """
        n_points = len(theta_grid)
        u_flat = u.reshape(-1)
        indices = torch.searchsorted(cdf_grid, u_flat).clamp(0, n_points - 1)
        return theta_grid[indices].reshape(u.shape)

    def sample(
        self, sample_shape: torch.Size | list[int] | tuple[int, ...] = torch.Size()
    ) -> torch.Tensor:
        """
        Draw samples via inverse transform sampling.

        :param sample_shape: Desired batch shape. Pass ``torch.Size([n])`` for
            *n* independent samples.
        :returns: Sampled tensor of shape ``(*sample_shape, event_size)``.
        """
        if not sample_shape:
            n_samples = 1
        elif isinstance(sample_shape, torch.Size):
            n_samples = sample_shape.numel()
        else:
            n_samples = len(sample_shape)

        theta_grid, cdf_grid = self._build_cdf_grid()
        u = self._sample_uniform_cdf(
            n_samples, cdf_min=cdf_grid[0], cdf_max=cdf_grid[-1]
        )
        samples = self._invert_cdf(u, theta_grid, cdf_grid)
        return samples.squeeze(0) if not sample_shape else samples
