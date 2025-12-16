from pathlib import Path

from mach3sbitools.file_io.root_file_handler import RootFileHandler
from mach3sbitools.file_io.file_handler_base import FileHandlerBase

__IMPLEMENTED_HANDLERS__ = {
    'root': RootFileHandler
}

def file_handler_factory(handler_name: str, input_file_name: Path, **kwargs)->FileHandlerBase:
    file_handler = __IMPLEMENTED_HANDLERS__.get(handler_name.lower())
    
    if file_handler is None:
        raise ValueError(f"Cannot find file io {handler_name} in {__IMPLEMENTED_HANDLERS__.keys()}")
    
    return file_handler(input_file_name, **kwargs)
