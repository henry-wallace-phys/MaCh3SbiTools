from mach3sbitools import ui
from mach3sbitools.sbi.mach3_prior import create_mach3_prior
from mach3sbitools.utils.device_handler import TorchDeviceHander

from functools import wraps
from pathlib import Path
import pickle
import torch
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt

import numpy as np

from sbi.neural_nets import posterior_nn
from sbi.analysis.plot import sbc_rank_plot
from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp


class MaCh3InferenceNotSetup(Exception):
    pass


class MaCh3SBIInterface:
    device_handler = TorchDeviceHander()
    
    def __init__(
        self, 
        mach3_interface, 
        inference_method, 
        n_rounds: int, 
        samples_per_round: int, 
        autosave_interval: int = -1, 
        output_file: Path = Path("model_output.pkl"),
        scaling: str = "none"
    ):
        """
        Args:
            mach3_interface: MaCh3 interface for simulation
            inference_method: SBI inference method (NPE, NLE, NRE, etc.)
            n_rounds: Number of training rounds
            samples_per_round: Samples per round
            autosave_interval: Save interval (-1 to disable)
            output_file: Output file path
            scaling: Parameter scaling method ("none", "standardise", "normalise")
        """
        self._mach3_interface = mach3_interface
        self._n_rounds = n_rounds
        self._autosave_interval = autosave_interval
        self._output_file = output_file
        self._config_file = self._mach3_interface.get_config_file()
        self._scaling = scaling
        
        # Create prior with scaling
        self._prior = create_mach3_prior(
            self._mach3_interface, 
            self.device_handler.device,
            scaling=scaling
        )
        
        self._posterior = self._proposal = self._prior
        self._samples_per_round = samples_per_round
        self._n_params = len(self._mach3_interface.get_parameter_names())
        self._inference = inference_method

    @property
    def parameter_names(self):
        return self._mach3_interface.get_parameter_names()

    @property
    def posterior(self):
        return self._posterior
    
    @property
    def prior(self):
        return self._prior

    @property
    def x0(self):
        return self.device_handler.to_tensor(self._mach3_interface.get_data_bins())
    
    @property
    def get_n_params(self):
        return self._n_params
    
    def sample(self, n_samples: int, **kwargs):
        """
        Sample from posterior in scaled space.
        Use sample_original() to get samples in original parameter space.
        """
        return self._posterior.sample((n_samples, ), **kwargs)
    
    def sample_original(self, n_samples: int, **kwargs):
        """
        Sample from posterior and transform to original parameter space.
        """
        scaled_samples = self.sample(n_samples, **kwargs)
        return self._prior.transform_to_original(scaled_samples)

    def get_x_vals(self, theta: torch.Tensor, is_scaled: bool = True):
        """
        Get simulation outputs for given parameters.
        
        Args:
            theta: Parameters (in scaled or original space)
            is_scaled: If True, theta is in scaled space and will be converted to original
        
        Returns:
            Tuple of (x_vals, theta_vals) both as tensors
        """
        # Convert to original space if needed
        if is_scaled and self._scaling != "none":
            theta_original = self._prior.transform_to_original(theta)
        else:
            theta_original = theta
        
        # Enforce boundary conditions by clipping to bounds
        theta_original = torch.clamp(
            theta_original,
            min=self._prior.lower_bounds,
            max=self._prior.upper_bounds
        )
        
        # Convert to numpy array on CPU for simulation
        theta_cpu = theta_original.cpu().numpy()
        valid_theta_scaled = []
        valid_theta_original = []
        valid_x = []
        
        for i, t in enumerate(tqdm(theta_cpu, desc="Getting MC Spectra from MaCh3")):
            try:
                sims = self._mach3_interface.simulate(t)
                valid_x.append(np.array(sims))
                valid_theta_original.append(np.array(t))
                # Store the corresponding scaled parameter
                if is_scaled:
                    valid_theta_scaled.append(theta[i].cpu().numpy())
                else:
                    # Need to scale it
                    valid_theta_scaled.append(
                        self._prior.transform_to_scaled(
                            self.device_handler.to_tensor(t)
                        ).cpu().numpy()
                    )
            except Exception:
                continue
        
        if not len(valid_theta_scaled):
            raise Exception("Proposal has failed, no valid values found!")
        
        # Return x and theta in SCALED space for SBI training
        return (
            self.device_handler.to_tensor(np.array(valid_x)), 
            self.device_handler.to_tensor(np.array(valid_theta_scaled))
        )
    
    def simulate(self, **kwargs):
        """
        Sample from proposal and get corresponding simulations.
        Returns samples in scaled space for SBI.
        """
        theta_scaled = self._proposal.sample((self._samples_per_round, ), **kwargs)
        x, theta_scaled = self.get_x_vals(theta_scaled, is_scaled=True)
        return x, theta_scaled
    
    def train(self, sampling_settings, training_settings):
        """
        Train the SBI model.
        
        Args:
            sampling_settings: Settings for posterior sampling
            training_settings: Settings for neural network training
        """
        self._proposal = self._prior
        
        if self._inference is None:
            raise MaCh3InferenceNotSetup("Inference method has not been setup yet!")
        
        print(f"Using scaling method: {self._scaling}")
        
        if self._scaling != "none":
            print("Prior scaling info:")
            for i, name in enumerate(self.parameter_names):
                print(f"  {name}: loc={self._prior.scale_loc[i]:.4f}, scale={self._prior.scale_scale[i]:.4f}")
        
        for t in tqdm(range(self._n_rounds)):
            self.training_iter(sampling_settings, training_settings)
            if self._autosave_interval > 0 and t % self._autosave_interval == 0:
                self.save(self._output_file)
    
    def training_iter(self, sampling_settings, training_settings):
        """Single training iteration."""
        x, theta = self.simulate()
        estim = self._inference.append_simulations(
            theta, x, proposal=self._proposal
        ).train(**training_settings, show_train_summary=True)
        self._posterior = self._inference.build_posterior(estim, **sampling_settings).set_default_x(self.x0)
        self._proposal = self._posterior
    
    def __getstate__(self):
        """Allow pickling."""
        state = self.__dict__.copy()
        state["_simulator"] = None
        return state

    def __setstate__(self, state):
        """Restore from pickle."""
        self.__dict__.update(state)
        self._mach3_interface = MaCh3Interface(self._config_file)

    def save(self, output: Path):
        """Save posterior to file."""
        output = Path(output)
        print(f"Saving to {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both the posterior and the scaling info
        save_dict = {
            'posterior': self._posterior,
            'scaling': self._scaling,
            'scale_loc': self._prior.scale_loc,
            'scale_scale': self._prior.scale_scale,
            'prior_info': self._prior.get_info()
        }
        
        with open(output, 'wb') as handle:
            pickle.dump(save_dict, handle)
    
    @staticmethod
    def load_posterior(input: Path):
        """Load posterior from file."""
        if not input.exists():
            raise FileNotFoundError(f"Cannot find file: {input}")
        
        with open(input, 'rb') as handle:
            save_dict = pickle.load(handle)
        
        return save_dict
    
    def get_sbc(self, out_folder: Path, n_simulations: int = 1000, num_posterior_samples: int = 1000):
        """
        Run Simulation-Based Calibration diagnostics.
        
        Args:
            out_folder: Output folder for plots
            n_simulations: Number of simulations for SBC
            num_posterior_samples: Number of posterior samples per simulation
        """
        # Sample from prior in scaled space
        thetas_scaled = self.prior.sample((n_simulations, ))
        x_sims, theta_sims_scaled = self.get_x_vals(thetas_scaled, is_scaled=True)
        
        ranks, dap_samples = run_sbc(
            theta_sims_scaled, x_sims, self._posterior,
            num_posterior_samples=num_posterior_samples,
            num_workers=8,
            use_batched_sampling=True
        )
        
        check_stats = check_sbc(
            ranks, theta_sims_scaled, dap_samples, 
            num_posterior_samples=num_posterior_samples
        )
        
        print(f"Kolmogorov-Smirnov p-values: {check_stats['ks_pvals'].numpy()}")
        print(f"C2ST accuracies (ranks): {check_stats['c2st_ranks'].numpy()}")
        print(f"C2ST accuracies (DAP): {check_stats['c2st_dap'].numpy()}")

        # Create plots
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)
        
        sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            parameter_labels=self._mach3_interface.get_parameter_names(),
            plot_type="hist",
            num_bins=None,
        )
        plt.savefig(out_folder / "sbc_rank_plot.pdf")
        plt.close()

        sbc_rank_plot(
            ranks, 
            num_posterior_samples, 
            plot_type="cdf", 
            parameter_labels=self._mach3_interface.get_parameter_names()
        )
        plt.savefig(out_folder / "sbc_rank_cdf_plot.pdf")
        plt.close()


def set_inference(inference_cls, **inference_kwargs):
    """
    Decorator to automatically set _inference in MaCh3SBIInterface subclasses.
    
    Args:
        inference_cls: The class of the inference method (e.g., NPE)
        inference_kwargs: Additional keyword arguments to pass to the inference class
    """
    def decorator(cls):
        orig_init = getattr(cls, "__init__", None)

        @wraps(orig_init)
        def __init__(
            self, 
            mach3_interface, 
            n_rounds, 
            samples_per_round, 
            autosave_interval=-1, 
            output_file=Path("model_output.pkl"),
            scaling="none",
            *args, 
            **kwargs
        ):
            # Call base class __init__ with inference=None
            super(cls, self).__init__(
                mach3_interface, 
                inference_method=None, 
                n_rounds=n_rounds,
                samples_per_round=samples_per_round,
                autosave_interval=autosave_interval,
                output_file=output_file,
                scaling=scaling
            )
            
            # Set the specific inference
            self._inference = inference_cls(
                self._prior, 
                device=self.device_handler.device, 
                show_progress_bars=False, 
                **inference_kwargs
            )
            
            # Call original __init__ if subclass had additional logic
            if orig_init and orig_init is not MaCh3SBIInterface.__init__:
                orig_init(self, mach3_interface, n_rounds, samples_per_round, 
                         autosave_interval, output_file, scaling, *args, **kwargs)

        cls.__init__ = __init__
        return cls
    return decorator


def set_inference_embedding(inference_cls, embedding_cls, nn_type="mdn", nn_args={}, **inference_kwargs):
    """
    Decorator to automatically set _inference with custom embedding in MaCh3SBIInterface subclasses.
    
    Args:
        inference_cls: The class of the inference method (e.g., NPE)
        embedding_cls: The class of the embedding network (e.g., FCEmbedding)
        nn_type: Type of density estimator ("mdn", "maf", "nsf")
        inference_kwargs: Additional keyword arguments to pass to the inference class
    """
    def decorator(cls):
        orig_init = getattr(cls, "__init__", None)

        @wraps(orig_init)
        def __init__(
            self, 
            mach3_interface, 
            n_rounds, 
            samples_per_round, 
            autosave_interval=-1, 
            output_file=Path("model_output.pkl"),
            scaling="none",
            *args, 
            **kwargs
        ):
            # Call base class __init__ with inference=None
            super(cls, self).__init__(
                mach3_interface, 
                inference_method=None, 
                n_rounds=n_rounds,
                samples_per_round=samples_per_round,
                autosave_interval=autosave_interval,
                output_file=output_file,
                scaling=scaling
            )
            
            # Create embedding
            embedding = embedding_cls(len(mach3_interface.get_mc_bins()))
            
            # Set the specific inference with embedding
            density_estimator = posterior_nn(nn_type, embedding_net=embedding, **nn_args)
            self._inference = inference_cls(
                self._prior, 
                density_estimator=density_estimator, 
                device=self.device_handler.device, 
                show_progress_bars=False, 
                **inference_kwargs
            )
            
            # Call original __init__ if subclass had additional logic
            if orig_init and orig_init is not MaCh3SBIInterface.__init__:
                orig_init(self, mach3_interface, n_rounds, samples_per_round, 
                         autosave_interval, output_file, scaling, *args, **kwargs)

        cls.__init__ = __init__
        return cls
    return decorator