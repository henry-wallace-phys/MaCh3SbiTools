from sbi.inference import NPE, NPE_A, NPE_B, FMPE, NLE
from sbi.utils import RestrictedPrior, get_density_thresholder
from sbi.inference.posteriors.posterior_parameters import VIPosteriorParameters

from mach3sbitools.mach3_interface.mach3_interface import MaCh3Interface
from mach3sbitools.sbi.sbi_mach3_interface import MaCh3SBIInterface

class FastEpsFree(MaCh3SBIInterface):
    def __init__(self, handler: MaCh3Interface, n_rounds: int, samples_per_round: int):
        super().__init__(handler, n_rounds, samples_per_round)
        self._inference = NPE_A(self._prior, show_progress_bars=False, device=self.device_handler.device)
        
    def training_iter(self, sampling_settings, training_settings):
        x, theta = self.simulate(**sampling_settings)
        
        final_round = iter == self._n_rounds - 1
                
        _ = self._inference.append_simulations(theta, x, proposal=self._proposal).train(final_round=final_round, **training_settings, show_train_summary=True)
        self._posterior = self._inference.build_posterior().set_default_x(self.x0)
        self._proposal = self._posterior

class AutomaticTransform(MaCh3SBIInterface):
    def __init__(self, handler: MaCh3Interface, n_rounds: int, samples_per_round: int):
        super().__init__(handler, n_rounds, samples_per_round)
        self._inference = NPE_B(self._prior, show_progress_bars=False, device=self.device_handler.device)

    def training_iter(self, sampling_settings, training_settings):
        x, theta = self.simulate(**sampling_settings)

        _ = self._inference.append_simulations(theta, x, proposal=self._proposal).train(**training_settings, show_train_summary=True)
        self._posterior = self._inference.build_posterior().set_default_x(self.x0)
        self._proposal = self._posterior

class DeistlerInference(MaCh3SBIInterface):
    def __init__(self, handler: MaCh3Interface, n_rounds: int, samples_per_round: int):
        super().__init__(handler, n_rounds, samples_per_round)
        self._inference = NPE(self._prior, show_progress_bars=False, device=self.device_handler.device)

    def training_iter(self, sampling_settings, training_settings):
        x, theta = self.simulate(**sampling_settings)

        _ = self._inference.append_simulations(theta, x, proposal=self._proposal).train(force_first_round_loss=True, show_train_summary=True, **training_settings)
        self._posterior = self._inference.build_posterior().set_default_x(self.x0)
        accept_reject_fn = get_density_thresholder(self._posterior, quantile=1e-4)
        self._proposal = RestrictedPrior(self._prior, accept_reject_fn, sample_with="sir", posterior=self._posterior, device=self.device_handler.device)

class Papamarkos(MaCh3SBIInterface):
    def __init__(self, handler: MaCh3Interface, n_rounds: int, samples_per_round: int):
        super().__init__(handler, n_rounds, samples_per_round)
        self._inference = NLE(self._prior, show_progress_bars=False, device=self.device_handler.device)

    def training_iter(self, sampling_settings, training_settings):
        x, theta = self.simulate(**sampling_settings)

        _ = self._inference.append_simulations(theta, x).train(**training_settings)
        self._posterior = self._inference.build_posterior(**sampling_settings)
        self._proposal = self._posterior.set_default_x(self.x0)
        
class Glockler(MaCh3SBIInterface):
    def __init__(self, handler: MaCh3Interface, n_rounds: int, samples_per_round: int):
        super().__init__(handler, n_rounds, samples_per_round)
        self._inference = NLE(self._prior, show_progress_bars=False, device=self.device_handler.device)
    
    def training_iter(self, sampling_settings, training_settings):
        x, theta = self.simulate()

        _ = self._inference.append_simulations(theta, x).train(**training_settings)
        self._posterior = self._inference.build_posterior(posterior_parameters=VIPosteriorParameters(vi_method="fKL"), **sampling_settings).set_default_x(self.x0)
        self._proposal = self._posterior.train()
