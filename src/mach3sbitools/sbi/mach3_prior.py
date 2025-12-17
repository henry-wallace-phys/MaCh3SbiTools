import torch
import torch
from torch.distributions import Distribution, constraints

from pyMaCh3Tutorial import MaCh3TutorialWrapper
from mach3sbitools.utils.device_handler import TorchDeviceHander
from mach3sbitools.mach3_interface.mach3_interface import MaCh3Interface

class MaCh3TorchPrior(Distribution):
    arg_constraints = {}
    support = constraints.real_vector

    def __init__(self, handler: MaCh3Interface, device_handler: TorchDeviceHander, validate_args=False):
        self._handler = handler
        
        self._device_handler = device_handler
        
        # FOR PICKLING
        self._config_file = self._handler.get_config_file()

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
        Compute log probability for theta under MaCh3 prior.
        
        Parameters:
            theta: torch.Tensor of shape (..., D)
        
        Returns:
            logp: torch.Tensor of shape (...,)
        """
        # --- 1. Check bounds per sample ---
        oob = (theta <= self.low) | (theta >= self.high)  # shape (..., D)
        oob_any = oob.any(dim=-1)  # shape (...,) True if any component is OOB

        # initialize logp with -inf for all samples
        logp = torch.full(theta.shape[:-1], float('-inf'), device=self.low.device)

        # --- 2. Only compute log_prob for in-bounds samples ---
        in_bounds_idx = ~oob_any
        if in_bounds_idx.any():
            theta_in_bounds = theta[in_bounds_idx]  # shape (N_in_bounds, D)

            # --- 3. Move to CPU for MaCh3 handler ---
            theta_cpu = theta_in_bounds.detach().cpu().numpy()

            # --- 4. Evaluate MaCh3 log_prior ---
            if theta_cpu.ndim == 2:
                logp_vals = [-1*self._handler.get_log_prior(p.tolist()) for p in theta_cpu]
                for i, l in enumerate(logp_vals):
                    if l < -1234567:
                        logp_vals[i] = float('-inf')
                        
                
            else:  # single vector
                logp_vals = -1*self._handler.get_log_prior(theta_cpu.tolist())
                if logp_vals<-1234567:
                    logp_vals = float('-inf')

            # --- 5. Move back to device ---
            logp[in_bounds_idx] = self._device_handler.to_tensor(logp_vals)

        return logp

    def __getstate__(self):
        state = self.__dict__.copy()
        # remove unpickleable C++ handler
        state["_handler"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # recreate C++ handler, singleton so it's the same across ALL processes!
        self._handler = MaCh3Interface(self._config_file)
