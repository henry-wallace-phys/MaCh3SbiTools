import mach3sbitools.sbi.sbi_mach3_fitters as sf
from mach3sbitools.sbi.sbi_mach3_interface import MaCh3SBIInterface
from mach3sbitools.mach3_interface.mach3_interface import MaCh3Interface
from pathlib import Path

__IMPLEMENTED_ALGORITHMS__ = {
    'fastepsfree': sf.FastEpsFree,
    'automatictransform': sf.AutomaticTransform,
    'deistlerinference': sf.DeistlerInference,
    'papamarkos': sf.Papamarkos,
    'glockler': sf.Glockler
    # 'flowmatching': sf.FlowMatching
}

def sbi_factory(fitter_name: str, file_handler: MaCh3Interface, n_rounds: int, samples_per_round: int, autosave_interval: int, output_file: Path)->MaCh3SBIInterface:
    sbi_fitter = __IMPLEMENTED_ALGORITHMS__.get(fitter_name.lower())
    if sbi_fitter is None:
        raise ValueError(f"Cannot find {sbi_fitter}, implemented algorithms are {__IMPLEMENTED_ALGORITHMS__.keys()}")
    
    return sbi_fitter(file_handler, n_rounds, samples_per_round, autosave_interval=autosave_interval, output_file=output_file)