import numpy as np
import torch
import torch.distributions


class CyclicalDistribution(torch.distributions.Distribution):
    def __init__(
        self,
        nominals: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ):
        """
        Technically the constructor is a bit pointlesss... but it's here
        in case I decide to go to bounds that aren't ±2pi
        """

        self.nominals = nominals
        self.device = nominals.device

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        if any(upper_bounds != 2 * torch.pi) or any(lower_bounds != -2 * torch.pi):
            raise NotImplementedError(
                "Cyclical prior not implemented for bounds that aren't [-2pi, 2pi]"
            )

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([len(nominals)]),
            validate_args=False,
        )

    @property
    def mean(self):
        return 0

    @property
    def variance(self) -> torch.Tensor:
        # Calculated from int(pdf*x2)dx using wolfram alpha
        return torch.Tensor([5.16947])

    def pdf(self, theta: torch.Tensor) -> torch.Tensor:
        in_bounds = (theta > self.lower_bounds) & (theta < self.upper_bounds)
        pdf = torch.zeros(theta.shape, dtype=torch.double)

        pdf[in_bounds] = (0.5 / torch.pi) * (
            torch.sin((theta[in_bounds] + 2 * torch.pi) / 4) ** 2
        )

        return pdf

    def cdf(self, theta: torch.Tensor) -> torch.Tensor:
        cdf = torch.zeros(theta.shape, dtype=torch.double)

        in_bounds = (theta > self.lower_bounds) & (theta < self.upper_bounds)
        cdf[in_bounds] = (0.5 / torch.pi) * (
            theta[in_bounds] / 2 + torch.sin(theta[in_bounds] / 2) + torch.pi
        )
        cdf[theta > self.upper_bounds] = 1.0
        return cdf

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        pdf = self.pdf(value)
        in_bounds = torch.abs(pdf) > 1e-8

        pdf[in_bounds] = torch.log(pdf[in_bounds])
        pdf[~in_bounds] = -np.inf

        return pdf

    def _build_cdf_grid(
        self, n_points: int = 10_000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute a lookup table of (theta, CDF(theta)) over the valid range."""
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
        """Sample u ~ Uniform(cdf_min, cdf_max), shaped (n_samples, event_size)."""
        event_size = len(self.nominals)
        u = torch.rand(n_samples, event_size, dtype=torch.double, device=self.device)
        return u * (cdf_max - cdf_min) + cdf_min

    def _invert_cdf(
        self, u: torch.Tensor, theta_grid: torch.Tensor, cdf_grid: torch.Tensor
    ) -> torch.Tensor:
        """Map uniform CDF samples back to theta via searchsorted on the lookup table."""
        n_points = len(theta_grid)
        u_flat = u.reshape(-1)
        indices = torch.searchsorted(cdf_grid, u_flat).clamp(0, n_points - 1)
        return theta_grid[indices].reshape(u.shape)

    def sample(
        self, sample_shape: torch.Size | list[int] | tuple[int, ...] = torch.Size()
    ) -> torch.Tensor:
        """Inverse transform sampling using a precomputed CDF lookup table."""
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
