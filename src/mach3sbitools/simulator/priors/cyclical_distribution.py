from torch import full_like, rand, Tensor, ones_like, log, Size, pi, sin, where, tensor, distributions, no_grad

class CyclicalDistribution(distributions.Distribution):
    def __init__(self, nominals: Tensor, lower_bounds: Tensor, upper_bounds: Tensor):
        '''
        Technically the constructor is a bit pointlesss... but it's here
        in case I decide to go to bounds that aren't ±2pi
        '''

        self.nominals = nominals

        self.device = nominals.device

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        if any(upper_bounds != 2 * pi) or any(lower_bounds != -2 * pi):
            raise NotImplementedError("Cyclical prior not implemented for bounds that aren't [-2pi, 2pi]")

        super().__init__(
            batch_shape=Size(),
            event_shape=Size([len(nominals)]),
            validate_args=False,
        )

    @staticmethod
    def _cdf(theta: Tensor) -> Tensor:
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
        u = (theta + 2.0 * pi) / 4.0
        return (u - sin(2.0 * u) / 2.0) / pi

    @staticmethod
    def _inv_cdf(p: Tensor, n_iter: int = 60) -> Tensor:
        """
        Invert the CDF via vectorised bisection.

        60 iterations give absolute accuracy < 4π / 2^60 ≈ 1 × 10⁻¹⁷.
        """
        lo = full_like(p, -2.0 * pi)
        hi = full_like(p, 2.0 * pi)

        for _ in range(n_iter):
            mid      = (lo + hi) * 0.5
            go_right = CyclicalDistribution._cdf(mid) < p
            lo       = where(go_right, mid, lo)
            hi       = where(go_right, hi, mid)

        return (lo + hi) * 0.5

    @property
    def variance(self) -> Tensor:
        """
        Variance computed analytically:

            ∫_{-2π}^{2π} θ² · p(θ) dθ  =  4π² − 8  ≈  31.48
        """
        return tensor(4.0 * pi ** 2 - 8.0, device=self.device)

    def sample(self, sample_shape=Size([])) -> Tensor:
        """Draw samples via exact inverse-CDF (no rejection needed)."""
        if not isinstance(sample_shape, Size):
            sample_shape = Size(
                [sample_shape] if isinstance(sample_shape, int) else sample_shape
            )
        with no_grad():
            u = rand(sample_shape, device=self.device)
            return self._inv_cdf(u)

    def rsample(self, sample_shape=Size([])) -> Tensor:
        """
        Reparameterised sample.

        The inverse-CDF transform is differentiable everywhere inside the
        open support, so gradients flow through correctly.
        """
        if not isinstance(sample_shape, Size):
            sample_shape = Size(
                [sample_shape] if isinstance(sample_shape, int) else sample_shape
            )
        u = rand(sample_shape, device=self.device)
        return self._inv_cdf(u)

    def log_prob(self, value: Tensor) -> Tensor:
        """
        Log probability under the cyclical prior.

            log p(θ) = −log(2π) + 2 · log|sin((θ + 2π)/4)|

        Returns -inf for θ outside (−2π, 2π) and at the endpoints where
        the density is zero.
        """
        if value.device.type != self.device:
            value = value.to(self.device)

        in_bounds = (value > self.lower_bounds) & (value < self.upper_bounds)

        sin_val  = sin((value + 2.0 * pi) / 4.0)
        # Substitute 1.0 outside bounds to avoid log(0) before the final mask
        safe_sin = where(in_bounds, sin_val.abs(), ones_like(sin_val))

        log_p = -log(2.0 * pi) + 2.0 * log(safe_sin)
        return where(in_bounds, log_p, tensor(float("-inf"), device=self.device))