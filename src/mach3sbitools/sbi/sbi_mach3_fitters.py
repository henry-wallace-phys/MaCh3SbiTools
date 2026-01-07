from sbi.inference import NPE, NPE_A, NPE_B, FMPE, NLE, NPSE
from sbi.utils import RestrictedPrior, get_density_thresholder
from sbi.inference.posteriors.posterior_parameters import VIPosteriorParameters
from sbi.neural_nets.embedding_nets import FCEmbedding


from mach3sbitools.sbi.sbi_mach3_interface import MaCh3SBIInterface, set_inference, set_inference_embedding
 
__NN_ARGS__={
    'num_bins': 20,
    'hidden_features': 256,
    'num_transforms': 20,
    'num_components': 50
}
 
'''
Simple classes:
'''
# NPE
@set_inference_embedding(NPE_B, FCEmbedding, nn_type='nsf', nn_args=__NN_ARGS__)
class FastSplinedMechanisticEmbedding(MaCh3SBIInterface): ...

# NPE
@set_inference_embedding(NPE_B, FCEmbedding, nn_type='maf', nn_args=__NN_ARGS__)
class FastMechanisticEmbedding(MaCh3SBIInterface): ...

@set_inference_embedding(NPE, FCEmbedding, nn_type='nsf', nn_args=__NN_ARGS__)
class AutomaticSplinedTransform(MaCh3SBIInterface): ...

@set_inference_embedding(NPE, FCEmbedding, nn_type='maf', nn_args=__NN_ARGS__)
class AutomaticTransform(MaCh3SBIInterface): ...

# NLE
@set_inference(NLE)
class SequentialNeuraLikelihood(MaCh3SBIInterface): ...

@set_inference(NPE_A)
class FastEpsFree(MaCh3SBIInterface):
    # Training iterations
    _iters = 0
    
    def training_iter(self, sampling_settings, training_settings):
        x, theta = self.simulate()
        self._iters+=1        
        final_round = self._iters == self._n_rounds     
        _ = self._inference.append_simulations(theta, x, proposal=self._proposal).train(final_round=final_round, **training_settings, show_train_summary=True)
        self._posterior = self._inference.build_posterior(**sampling_settings).set_default_x(self.x0)
        self._proposal = self._posterior

@set_inference(NPE)
class TruncatedProposal(MaCh3SBIInterface):
    def training_iter(self, sampling_settings, training_settings):
        training_settings['force_first_round_loss'] = True
        super().training_iter(sampling_settings, training_settings)
        accept_reject_fn = get_density_thresholder(self._posterior, quantile=1e-4)
        self._proposal = RestrictedPrior(self._prior, accept_reject_fn, sample_with="sir", posterior=self._posterior, device=self.device_handler.device)


@set_inference(NPSE, sde_type='ve')
class NeuralPosteriorScoreEstimation(MaCh3SBIInterface):
    def train(self, sampling_settings, training_settings):
        if self._n_rounds>1:
            raise ValueError("Neural Posterior Score Estimation currently only supports single round training!")
        super().train(sampling_settings, training_settings)

@set_inference(FMPE)
class FlowMatching(MaCh3SBIInterface):    
    def train(self, sampling_settings, training_settings):
        if self._n_rounds>1:
            raise ValueError("Flow Matching currently only supports single round training!")
        super().train(sampling_settings, training_settings)
        
# Neural Likelihoods
@set_inference(NLE)
class VariationalLikelihoodEstimator(MaCh3SBIInterface):
    def training_iter(self, sampling_settings, training_settings):
        sampling_settings['posterior_parameters'] = VIPosteriorParameters(vi_method="fKL")
        super().training_iter(sampling_settings, training_settings)
        self._proposal = self._posterior.train()