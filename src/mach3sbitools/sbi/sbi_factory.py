import mach3sbitools.sbi.sbi_mach3_fitters as sf
from mach3sbitools.sbi.sbi_mach3_interface import MaCh3SBIInterface
from pathlib import Path
from mach3sbitools.sbi.sbi_mach3_interface import MaCh3SBIInterface, set_inference, set_inference_embedding


__IMPLEMENTED_ALGORITHMS__ = {
    # NPE
    'fast_eps_free': sf.FastEpsFree,
    'automatic_transform': sf.AutomaticTransform,
    'splined_automatic_transform': sf.AutomaticSplinedTransform,
    'mechanistic_embedding': sf.FastMechanisticEmbedding,
    'splined_mechanistic_embedding': sf.FastSplinedMechanisticEmbedding,
    'truncated_proposal': sf.TruncatedProposal,
    'flow_matching': sf.FlowMatching,
    # NLE
    'neural_posterior_score_estimation': sf.NeuralPosteriorScoreEstimation,
    'sequtential_neural_likelihood': sf.SequentialNeuraLikelihood,
    'variational_likelihood_estimator': sf.VariationalLikelihoodEstimator,
}

def sbi_factory(fitter_name: str, mach3_interface, n_rounds: int, samples_per_round: int, autosave_interval: int, output_file: Path)->MaCh3SBIInterface:
    sbi_fitter = __IMPLEMENTED_ALGORITHMS__.get(fitter_name.lower())
    if sbi_fitter is None:
        raise ValueError(f"Cannot find {sbi_fitter}, implemented algorithms are {__IMPLEMENTED_ALGORITHMS__.keys()}")
    
    return sbi_fitter(mach3_interface, n_rounds, samples_per_round, autosave_interval=autosave_interval, output_file=output_file)