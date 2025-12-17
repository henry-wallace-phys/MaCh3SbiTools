import torch
import torch
from torch.distributions import Distribution, constraints

from pyMaCh3Tutorial import MaCh3TutorialWrapper
from mach3sbitools.utils.device_handler import TorchDeviceHander

class MaCh3TorchPrior(Distribution):
    arg_constraints = {}
    support = constraints.real_vector

    def __init__(self, handler: MaCh3TutorialWrapper, device_handler: TorchDeviceHander, validate_args=False):
        self._handler = handler
        self._device_handler = device_handler

        low, high = handler.get_bounds()

        self.low = self._device_handler.to_tensor(low)
        self.high = self._device_handler.to_tensor(high)

        self._dim = self.low.shape[0]

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([self._dim]),
            validate_args=validate_args,
        )

    # ------------------------------------------------------------
    # Sampling (uniform inside bounds)
    # ------------------------------------------------------------
    def sample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.event_shape
        u = torch.rand(shape, device=self.low.device)
        return self.low + u * (self.high - self.low)

    # ------------------------------------------------------------
    # Log probability (delegates to MaCh3)
    # ------------------------------------------------------------
    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """
        theta shape: (..., D)
        returns: (...,)
        """

        # bounds check (vectorized, on device)
        oob = (theta <= self.low) | (theta >= self.high)
        if oob.any():
            return torch.full(
                theta.shape[:-1],
                -torch.inf,
                device=self.low.device
            )

        # ---- move to CPU for MaCh3 ----
        theta_cpu = theta.detach().cpu().numpy()

        # batched evaluation
        if theta_cpu.ndim == 2:
            logp = [
                self._handler.get_log_prior(p.tolist())
                for p in theta_cpu
            ]
        else:
            logp = self._handler.get_log_prior(theta_cpu.tolist())

        # ---- back to torch device ----
        print(logp)
        return self._device_handler.to_tensor(logp)
