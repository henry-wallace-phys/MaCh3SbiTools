import mach3sbitools.sbi.sbi_fitters as sf
from mach3sbitools.file_io.file_handler_base import FileHandlerBase
from mach3sbitools.sbi.sbi_interface import SbiInterface


__IMPLEMENTED_ALGORITHMS__ = {
    'fastepsfree': sf.FastEpsFree,
    'automatictransform': sf.AutomaticTransform,
    'flowmatching': sf.FlowMatching
}

def sbi_factory( fitter_name: str, file_handler: FileHandlerBase, n_rounds: int, prior)->SbiInterface:
    sbi_fitter = __IMPLEMENTED_ALGORITHMS__.get(fitter_name.lower())
    if sbi_fitter is None:
        raise ValueError(f"Cannot find {sbi_fitter}, implemented algorithms are {__IMPLEMENTED_ALGORITHMS__.keys()}")
    
    return sbi_fitter(file_handler, n_rounds, prior)