"""
Helper module to create SBI-compatible priors from MaCh3 parameter information
with parameter scaling for improved neural network training using PyTorch transforms.
"""

import torch
from typing import List, Tuple, Optional, Literal
from torch.distributions import Independent, Normal, Uniform, TransformedDistribution
from torch.distributions.transforms import AffineTransform


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
        parameter_names: Optional[List[str]] = None,
        device: str = "cpu",
        scaling: Literal["none", "standardise", "normalise"] = "none"
    ):
        """
        Args:
            nominals: List of nominal/central values for each parameter
            errors: List of errors (1-sigma) for Gaussian priors, or -1 for flat priors
            bounds: Tuple of (lower_bounds, upper_bounds) for each parameter
            parameter_names: Optional list of parameter names for reference
            device: Device to place tensors on ('cpu', 'cuda', 'mps')
            scaling: Scaling method to use:
                - "none": No scaling (original parameter space)
                - "standardise": Scale to mean=0, std=1 (recommended for mixed priors)
                - "normalise": Scale to [0, 1] based on bounds
        """
        self.device = device
        self.nominals = torch.tensor(nominals, dtype=torch.float32, device=device)
        self.errors = torch.tensor(errors, dtype=torch.float32, device=device)
        self.lower_bounds = torch.tensor(bounds[0], dtype=torch.float32, device=device)
        self.upper_bounds = torch.tensor(bounds[1], dtype=torch.float32, device=device)
        self.parameter_names = parameter_names or [f"param_{i}" for i in range(len(nominals))]
        
        self.n_params = len(nominals)
        self.scaling_method = scaling
        
        # Identify which parameters have Gaussian vs flat priors
        self.gaussian_mask = self.errors > 0
        self.flat_mask = self.errors < 0
        
        self.n_gaussian = self.gaussian_mask.sum().item()
        self.n_flat = self.flat_mask.sum().item()
        
        if self.n_gaussian + self.n_flat != self.n_params:
            raise ValueError("Each parameter must be either Gaussian (error > 0) or flat (error = -1)")
        
        # Compute scaling parameters
        self._compute_scaling()
        
        # Build the base prior distribution (in original space)
        self._build_base_prior()
        
        # Apply scaling transform if needed
        if self.scaling_method != "none":
            # Create inverse transform: scaled = (original - loc) / scale
            # For TransformedDistribution, we need the forward transform: original = scaled * scale + loc
            # So we create the inverse and then use TransformedDistribution
            self._prior = TransformedDistribution(
                self._base_prior,
                AffineTransform(loc=-self.scale_loc / self.scale_scale, scale=1.0 / self.scale_scale)
            )
        else:
            self._prior = self._base_prior
        
        # Initialize parent class
        super().__init__(
            batch_shape=self._prior.batch_shape,
            event_shape=self._prior.event_shape,
            validate_args=False
        )
    
    def _compute_scaling(self):
        """Compute scaling parameters for each dimension."""
        self.scale_loc = torch.zeros(self.n_params, device=self.device)
        self.scale_scale = torch.ones(self.n_params, device=self.device)
        
        if self.scaling_method == "none":
            # No scaling
            pass
            
        elif self.scaling_method == "standardise":
            # Scale to mean=0, std=1
            # For Gaussian parameters: use nominal as mean, error as std
            # For flat parameters: use midpoint as mean, range/sqrt(12) as std (uniform std)
            
            for i in range(self.n_params):
                if self.gaussian_mask[i]:
                    self.scale_loc[i] = self.nominals[i]
                    self.scale_scale[i] = self.errors[i]
                else:
                    # Uniform distribution
                    midpoint = (self.lower_bounds[i] + self.upper_bounds[i]) / 2
                    # Standard deviation of uniform distribution: (b-a)/sqrt(12)
                    std = (self.upper_bounds[i] - self.lower_bounds[i]) / (2 * torch.sqrt(torch.tensor(3.0, device=self.device)))
                    self.scale_loc[i] = midpoint
                    self.scale_scale[i] = std
                    
        elif self.scaling_method == "normalise":
            # Scale to [0, 1] based on bounds
            self.scale_loc = self.lower_bounds
            self.scale_scale = self.upper_bounds - self.lower_bounds
            
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def _build_base_prior(self):
        """Construct the base prior distribution in original parameter space."""
        
        if self.n_flat == 0:
            # All Gaussian
            self._base_prior = Independent(
                Normal(loc=self.nominals, scale=self.errors),
                1
            )
            self.prior_type = "gaussian"
            
        elif self.n_gaussian == 0:
            # All flat
            self._base_prior = Independent(
                Uniform(low=self.lower_bounds, high=self.upper_bounds),
                1
            )
            self.prior_type = "uniform"
            
        else:
            # Mixed prior
            self._base_prior = MixedPrior(
                nominals=self.nominals,
                errors=self.errors,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                gaussian_mask=self.gaussian_mask,
                flat_mask=self.flat_mask,
                device=self.device
            )
            self.prior_type = "mixed"
    
    def transform_to_original(self, scaled_params: torch.Tensor) -> torch.Tensor:
        """
        Transform parameters from scaled space back to original space.
        
        Args:
            scaled_params: Parameters in scaled space (shape: [..., n_params])
            
        Returns:
            Parameters in original space (shape: [..., n_params])
        """
        if self.scaling_method == "none":
            return scaled_params
        return scaled_params * self.scale_scale + self.scale_loc
    
    def transform_to_scaled(self, original_params: torch.Tensor) -> torch.Tensor:
        """
        Transform parameters from original space to scaled space.
        
        Args:
            original_params: Parameters in original space (shape: [..., n_params])
            
        Returns:
            Parameters in scaled space (shape: [..., n_params])
        """
        if self.scaling_method == "none":
            return original_params
        return (original_params - self.scale_loc) / self.scale_scale
    
    @property
    def support(self):
        """Return the support of the distribution."""
        return self._prior.support
    
    def rsample(self, sample_shape=torch.Size()):
        """Sample with reparameterization (required by some SBI methods)."""
        return self._prior.rsample(sample_shape)
    
    def sample(self, sample_shape=torch.Size([])):
        """Sample from the prior (in scaled space if scaling is enabled)."""
        return self._prior.sample(sample_shape)
    
    def log_prob(self, value):
        """Compute log probability of value under the prior (value in scaled space if scaling is enabled)."""
        return self._prior.log_prob(value)
    
    def get_info(self):
        """Get information about the prior."""
        info = {
            "n_params": self.n_params,
            "n_gaussian": self.n_gaussian,
            "n_flat": self.n_flat,
            "prior_type": self.prior_type,
            "scaling_method": self.scaling_method,
            "parameter_names": self.parameter_names
        }
        
        # Add scaling info
        if self.scaling_method != "none":
            info["scaling"] = {
                "scale_loc": self.scale_loc.cpu().numpy().tolist(),
                "scale_scale": self.scale_scale.cpu().numpy().tolist()
            }
        
        # Add details for each parameter
        for i, name in enumerate(self.parameter_names):
            param_info = {}
            if self.gaussian_mask[i]:
                param_info["type"] = "gaussian"
                param_info["nominal"] = self.nominals[i].item()
                param_info["error"] = self.errors[i].item()
                param_info["bounds"] = (self.lower_bounds[i].item(), self.upper_bounds[i].item())
            else:
                param_info["type"] = "flat"
                param_info["bounds"] = (self.lower_bounds[i].item(), self.upper_bounds[i].item())
            
            if self.scaling_method != "none":
                param_info["scale_loc"] = self.scale_loc[i].item()
                param_info["scale_scale"] = self.scale_scale[i].item()
                
            info[name] = param_info
        
        return info
    
    def check_bounds(self, params: torch.Tensor, is_scaled: bool = False) -> torch.Tensor:
        """
        Check if parameters are within bounds.
        
        Args:
            params: Parameters to check (shape: [..., n_params])
            is_scaled: If True, params are in scaled space
            
        Returns:
            Boolean tensor indicating which samples are in bounds (shape: [...])
        """
        if is_scaled and self.scaling_method != "none":
            params = self.transform_to_original(params)
        
        in_bounds = (params >= self.lower_bounds) & (params <= self.upper_bounds)
        return in_bounds.all(dim=-1)


class MixedPrior(torch.distributions.Distribution):
    """
    Custom distribution for mixed Gaussian and flat priors in original parameter space.
    
    This is the base distribution before any scaling is applied.
    """
    
    def __init__(
        self,
        nominals: torch.Tensor,
        errors: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        gaussian_mask: torch.Tensor,
        flat_mask: torch.Tensor,
        device: str = "cpu"
    ):
        batch_shape = torch.Size()
        event_shape = torch.Size([len(nominals)])
        
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=False)
        
        self.device = device
        self.nominals = nominals.to(device)
        self.errors = errors.to(device)
        self.lower_bounds = lower_bounds.to(device)
        self.upper_bounds = upper_bounds.to(device)
        self.gaussian_mask = gaussian_mask.to(device)
        self.flat_mask = flat_mask.to(device)
        
        self.gaussian_indices = torch.where(gaussian_mask)[0]
        self.flat_indices = torch.where(flat_mask)[0]
        
        # Create component distributions
        self.component_dists = []
        
        for i in range(len(nominals)):
            if gaussian_mask[i]:
                dist = Normal(loc=nominals[i], scale=errors[i])
            else:
                dist = Uniform(low=lower_bounds[i], high=upper_bounds[i])
            self.component_dists.append(dist)
    
    @property
    def support(self):
        """Return the support of the distribution."""
        from torch.distributions import constraints
        return constraints.independent(
            constraints.interval(self.lower_bounds, self.upper_bounds),
            1
        )
    
    @property
    def mean(self):
        """Return the mean of the distribution."""
        means = torch.zeros_like(self.nominals, device=self.device)
        means[self.gaussian_mask] = self.nominals[self.gaussian_mask]
        means[self.flat_mask] = (self.lower_bounds[self.flat_mask] + self.upper_bounds[self.flat_mask]) / 2
        return means
    
    @property
    def variance(self):
        """Return the variance of the distribution."""
        variances = torch.zeros_like(self.nominals, device=self.device)
        if len(self.gaussian_indices) > 0:
            variances[self.gaussian_mask] = self.errors[self.gaussian_mask] ** 2
        if len(self.flat_indices) > 0:
            variances[self.flat_mask] = (self.upper_bounds[self.flat_mask] - self.lower_bounds[self.flat_mask]) ** 2 / 12
        return variances
    
    def sample(self, sample_shape=torch.Size([])):
        """Sample from the mixed prior."""
        if not isinstance(sample_shape, torch.Size):
            if isinstance(sample_shape, (int, tuple, list)):
                sample_shape = torch.Size([sample_shape] if isinstance(sample_shape, int) else sample_shape)
            else:
                sample_shape = torch.Size([])
        
        samples = torch.zeros(sample_shape + (len(self.nominals),), device=self.device)
        
        for i, dist in enumerate(self.component_dists):
            param_samples = dist.sample(sample_shape)
            
            # Clip Gaussian parameters to bounds (uniform already respects bounds)
            if self.gaussian_mask[i]:
                param_samples = torch.clamp(param_samples, self.lower_bounds[i], self.upper_bounds[i])
            
            samples[..., i] = param_samples
        
        return samples
    
    def rsample(self, sample_shape=torch.Size([])):
        """Sample with reparameterization trick."""
        if not isinstance(sample_shape, torch.Size):
            if isinstance(sample_shape, (int, tuple, list)):
                sample_shape = torch.Size([sample_shape] if isinstance(sample_shape, int) else sample_shape)
            else:
                sample_shape = torch.Size([])
        
        samples = torch.zeros(sample_shape + (len(self.nominals),), device=self.device)
        
        for i, dist in enumerate(self.component_dists):
            if hasattr(dist, 'rsample'):
                param_samples = dist.rsample(sample_shape)
            else:
                param_samples = dist.sample(sample_shape)
            
            # Clip Gaussian parameters to bounds (uniform already respects bounds)
            if self.gaussian_mask[i]:
                param_samples = torch.clamp(param_samples, self.lower_bounds[i], self.upper_bounds[i])
            
            samples[..., i] = param_samples
        
        return samples
    
    def log_prob(self, value):
        """Compute log probability of value under the mixed prior."""
        if value.device.type != self.device:
            value = value.to(self.device)
        
        log_prob = torch.zeros(value.shape[:-1], device=self.device)
        
        for i, dist in enumerate(self.component_dists):
            param_values = value[..., i]
            param_log_prob = dist.log_prob(param_values)
            
            # For Gaussian parameters, handle bounds
            if self.gaussian_mask[i]:
                param_log_prob = torch.where(
                    (param_values >= self.lower_bounds[i]) & (param_values <= self.upper_bounds[i]),
                    param_log_prob,
                    torch.tensor(float('-inf'), device=self.device)
                )
            
            log_prob = log_prob + param_log_prob
        
        return log_prob


def create_mach3_prior(
    wrapper, 
    device: str = "cpu", 
    scaling: Literal["none", "standardise", "normalise"] = "none"
) -> MaCh3Prior:
    """
    Convenience function to create a prior from a MaCh3DUNEWrapper instance.
    
    Args:
        wrapper: Instance of MaCh3DUNEWrapper
        device: Device to place tensors on ('cpu', 'cuda', 'mps')
        scaling: Scaling method - "none", "standardise" (default), or "normalise"
        
    Returns:
        MaCh3Prior object compatible with SBI
    """
    nominals, errors = wrapper.get_nominal_error()
    bounds = wrapper.get_bounds()
    names = wrapper.get_parameter_names()
    
    return MaCh3Prior(
        nominals=nominals,
        errors=errors,
        bounds=bounds,
        parameter_names=names,
        device=device,
        scaling=scaling
    )