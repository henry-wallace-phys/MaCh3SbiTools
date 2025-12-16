from sbi.inference import NPE, NPE_A, NPE_B, FMPE

from pyMaCh3Tutorial import MaCh3TutorialWrapper
from mach3sbitools.sbi.sbi_mach3_interface import MaCh3SBIInterface

class FastEpsFree(MaCh3SBIInterface):
    def __init__(self, handler: MaCh3TutorialWrapper, n_rounds: int, samples_per_round: int, prior):
        super().__init__(handler, n_rounds, samples_per_round, prior)
        self._inference = NPE_A(prior, device=self.device_handler.device)
        self._proposal = None
    
    def train(self, sampling_settings, training_settings):
        self._proposal = self._prior
        super().train(sampling_settings, training_settings)
        
    def training_iter(self, sampling_settings, training_settings):
        x, theta = self.simulate(**sampling_settings)
        
        final_round = iter == self._n_rounds - 1
                
        _ = self._inference.append_simulations(theta, x, proposal=self._proposal).train(final_round=final_round, **training_settings)
        self._posterior = self._inference.build_posterior().set_default_x(self.x0)
        self._proposal = self._posterior

class AutomaticTransform(MaCh3SBIInterface):
    def __init__(self, handler: MaCh3TutorialWrapper, n_rounds: int, samples_per_round: int, prior):
        self._inference = NPE_B(prior, device=self.device_handler.device)
        super().__init__(handler, n_rounds, samples_per_round, prior)
        self._proposal = None

    def train(self, sampling_settings, training_settings):
        self._proposal = self._prior
        super().train(sampling_settings, training_settings)


    def training_iter(self, sampling_settings, training_settings):
        x, theta = self.simulate(**sampling_settings)

        _ = self._inference.append_simulations(theta, x, proposal=self._proposal).train(**training_settings)
        self._posterior = self._inference.build_posterior().set_default_x(self.x0)
        self._proposal = self._posterior

