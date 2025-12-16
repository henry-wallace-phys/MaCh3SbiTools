from sbi.inference import NPE, NPE_A, FMPE

from mach3sbitools.sbi.sbi_interface import SbiInterface
from mach3sbitools.file_io.file_handler_base import FileHandlerBase

class FastEpsFree(SbiInterface):
    def __init__(self, file_handler: FileHandlerBase, n_rounds: int, prior):
        super().__init__(file_handler, n_rounds, prior)
        self._inference = NPE_A(prior, device=self.device_handler.device)
        self._proposal = None
    
    def train(self, **kwargs):
        self._proposal = self._prior
        super().train(**kwargs)
        
    def training_iter(self, iter, file_args, **kwargs):
        self.load_x_theta(iter, **file_args)
        
        training_kwargs = kwargs.get("TrainingSettings", {})
        final_round = iter == self._n_rounds - 1
        
        if self.theta is None or self.x is None:
            raise ValueError("Input file has not loaded x or theta!")
        
        _ = self.inference.append_simulations(self.theta, self.x, proposal=self._proposal).train(final_round=final_round, **training_kwargs)
        self._posterior = self._inference.build_posterior().set_default_x(self.x0)
        
        self._proposal = self._posterior
        
class AutomaticTransform(SbiInterface):
    def __init__(self, file_handler: FileHandlerBase, n_rounds: int, prior):
        super().__init__(file_handler, n_rounds, prior)
        self._inference = NPE_B(prior, device=self.device_handler.device)
        self._proposal = None

    def train(self, **kwargs):
        self._proposal = self._prior
        super().train(**kwargs)


    def training_iter(self, iter, file_args, **kwargs):
        self.load_x_theta(iter, **file_args)

        training_kwargs = kwargs.get("TrainingSettings", {})

        _ = self.inference.append_simulations(self.theta, self.x, proposal=self._proposal).train(**training_kwargs)
        self._posterior = self._inference.build_posterior().set_default_x(self.x0)
        self._proposal = self._posterior


class FlowMatching(SbiInterface):
    def __init__(self, file_handler: FileHandlerBase, n_rounds: int, prior):
        super().__init__(file_handler, n_rounds, prior)
        self._inference = FMPE(prior, device=self.device_handler.device)
        self._proposal = None

    def training_iter(self, iter: int, file_args: dict, **kwargs):
        self.load_x_theta(iter, **file_args)

        training_kwargs = kwargs.get("TrainingSettings", {})
        
        self._inference.append_simulations(self.theta, self.x).train(**training_kwargs)
        self._posterior = self._inference.build_posterior()
        
        
