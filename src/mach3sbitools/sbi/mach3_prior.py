from sbi.utils import BoxUniform
from torch.distributions import Independent, Normal, Distribution
import torch

from mach3sbitools.utils.device_handler import TorchDeviceHander
from mach3sbitools.mach3_interface.mach3_interface import MaCh3Interface

class MaCh3Prior(Distribution):
    """
    A prior that adapts based on simulator interface specifications.
    
    - If prior_sigma >= 0: Uses a Gaussian (Normal) prior with the given mean and std
    - If prior_sigma < 0: Uses a flat (Uniform) prior over the bounds
    
    Parameters
    ----------
    simulator : object
        Simulator object with get_bounds() and get_nominal_error() methods
    """
    
    def __init__(self, simulator: MaCh3Interface):
        self._simulator = simulator
        self._config_file = simulator.get_config_file()
        self.device_handler = TorchDeviceHander()
        
        # Get bounds from simulator
        lower_bounds, upper_bounds = simulator.get_bounds()
        self.lower_bounds = self.device_handler.to_tensor(lower_bounds).float()
        self.upper_bounds = self.device_handler.to_tensor(upper_bounds).float()
        
        # Get nominal error (prior specification)
        prior_mu, prior_sigma = simulator.get_nominal_error()
        self.prior_mu = self.device_handler.to_tensor(prior_mu).float()
        self.prior_sigma = self.device_handler.to_tensor(prior_sigma).float()
        
        # Determine prior type and create appropriate distribution
        if torch.all(self.prior_sigma >= 0):
            # Gaussian prior
            self.prior_type = "gaussian"
            self._distribution = Independent(
                Normal(loc=self.prior_mu, scale=self.prior_sigma),
                reinterpreted_batch_ndims=1
            )
        else:
            # Flat (uniform) prior
            self.prior_type = "uniform"
            self._distribution = BoxUniform(
                low=self.lower_bounds,
                high=self.upper_bounds
            )
        
        # Initialize the parent Distribution class
        super().__init__(
            batch_shape=self._distribution.batch_shape,
            event_shape=self._distribution.event_shape,
            validate_args=False
        )
    
    def sample(self, sample_shape=torch.Size()):
        """Sample from the prior."""
        samples = self._distribution.sample(sample_shape)
        
        # For Gaussian prior, clip to bounds
        if self.prior_type == "gaussian":
            samples = torch.max(samples, self.lower_bounds)
            samples = torch.min(samples, self.upper_bounds)
        
        return samples
    
    def log_prob(self, value):
        """Compute log probability of value under the prior."""
        # Check if value is within bounds
        within_bounds = torch.all(
            (value >= self.lower_bounds) & (value <= self.upper_bounds),
            dim=-1
        )
        
        log_prob = self._distribution.log_prob(value)
        
        # For values outside bounds, set log_prob to -inf
        neg_inf = self.device_handler.to_tensor(-float('inf'))
        log_prob = torch.where(within_bounds, log_prob, neg_inf)
        
        return log_prob
    
    @property
    def mean(self):
        """Return the mean of the prior."""
        if self.prior_type == "gaussian":
            # Clip mean to bounds
            return torch.max(torch.min(self.prior_mu, self.upper_bounds), self.lower_bounds)
        else:
            # For uniform, return midpoint
            return (self.lower_bounds + self.upper_bounds) / 2
    
    @property
    def device(self):
        """Return the device where tensors are stored."""
        return self.device_handler.device
    
    def __repr__(self):
        return (f"AdaptivePrior(type={self.prior_type}, "
                f"bounds=[{self.lower_bounds.tolist()}, {self.upper_bounds.tolist()}])")


    # Allows us to pickle the file!
    def __getstate__(self):
        state = self.__dict__.copy()
        # remove unpickleable C++ handler
        state["_simulator"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # recreate C++ handler, singleton so it's the same across ALL processes!
        self._simulator = MaCh3Interface(self._config_file)
