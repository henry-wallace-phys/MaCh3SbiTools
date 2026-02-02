import importlib

class MaCh3ImportError(Exception):
    """Custom exception for MaCh3 import errors."""
    pass

class MaCh3ConfigError(Exception):
    """Custom exception for MaCh3 configuration errors."""
    pass

__SUPPORTED__MACH3S__ = ["DUNE", "Tutorial"]

def get_mach3_wrapper(mach3_name: str, config_file: Path) -> object:
    """
    Dynamically imports and returns the MaCh3 wrapper class for the specified mach3_name.
    """
    if mach3_name not in __SUPPORTED__MACH3S__:
        raise MaCh3ImportError(f"Unsupported MaCh3: {mach3_name}. Currently supported: {__SUPPORTED__MACH3S__}")
    
    if not config_file.exists():
        raise MaCh3ConfigError(f"Config file does not exist: {config_file}")
    
    try:
        module = importlib.import_module(f"pyMaCh3{mach3_name}")
        return getattr(module, f"MaCh3{mach3_name}Wrapper")(str(config_file))
    except ImportError as e:
        raise MaCh3ImportError(f"Failed to import {mach3_name}: {e}")
