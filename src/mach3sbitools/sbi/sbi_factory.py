import mach3sbitools.sbi.sbi_mach3_fitters as sf
from mach3sbitools.sbi.sbi_interface import SbiInterface
from pyMaCh3Tutorial import MaCh3TutorialWrapper


__IMPLEMENTED_ALGORITHMS__ = {
    'fastepsfree': sf.FastEpsFree,
    'automatictransform': sf.AutomaticTransform,
    # 'flowmatching': sf.FlowMatching
}

def sbi_factory(fitter_name: str, file_handler: MaCh3TutorialWrapper, n_rounds: int, samples_per_round: int, prior)->SbiInterface:
    sbi_fitter = __IMPLEMENTED_ALGORITHMS__.get(fitter_name.lower())
    if sbi_fitter is None:
        raise ValueError(f"Cannot find {sbi_fitter}, implemented algorithms are {__IMPLEMENTED_ALGORITHMS__.keys()}")
    
    return sbi_fitter(file_handler, n_rounds, samples_per_round, prior)