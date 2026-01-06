dune_wrapper_found=False
tutorial_wrapper_found=False
try:
    from pyMaCh3DUNE import MaCh3DUNEWrapper
    print("Found MaCh3 DUNE instance, using that!")
    dune_wrapper_found = True
except Exception:
    print("Couldn't find DUNE instance")    
try:
    from pyMaCh3Tutorial import MaCh3TutorialWrapper
    print("Found MaCh3 Tutorial instance, using that!")
    tutorial_wrapper_found = True
except Exception:
    print("Couldn't find Tutorial instance")    
    
from threading import Lock

'''
Interface classes for MaCh3 wrappers, with per-process singleton caching.
'''

if dune_wrapper_found:
    class MaCh3DUNEInterface(MaCh3DUNEWrapper):
        """
        Per-process singleton cache for MaCh3Wrapper objects,
        keyed by config file. Inherits from MaCh3[DUNE/Tutorial]Wrapper (these are ducktyped!).
        """
        _cache = {}
        _lock = Lock()

        def __new__(cls, config_file: str, *args, **kwargs):
            # Fast path
            if config_file in cls._cache:
                return cls._cache[config_file]

            with cls._lock:
                if config_file not in cls._cache:
                    # Create a new instance
                    instance = super().__new__(cls)
                    instance._init_singleton(config_file, *args, **kwargs)
                    cls._cache[config_file] = instance

            return cls._cache[config_file]

        def _init_singleton(self, config_file: str, *args, **kwargs):
            # Call the original MaCh3TutorialWrapper initializer
            super().__init__(config_file)
            self._config_file = config_file  # store for reference

        def get_config_file(self):
            return self._config_file
else:
    class MaCh3DUNEInterface:
        def __init__(self, *args, kwargs):
            raise Exception("MACH3 DUNE NOT SET UP CORRECTLY")

if tutorial_wrapper_found:
    class MaCh3TutorialInterface(MaCh3TutorialWrapper):
        """
        Per-process singleton cache for MaCh3Wrapper objects,
        keyed by config file. Inherits from MaCh3[DUNE/Tutorial]Wrapper (these are ducktyped!).
        """
        _cache = {}
        _lock = Lock()

        def __new__(cls, config_file: str, *args, **kwargs):
            # Fast path
            if config_file in cls._cache:
                return cls._cache[config_file]

            with cls._lock:
                if config_file not in cls._cache:
                    # Create a new instance
                    instance = super().__new__(cls)
                    instance._init_singleton(config_file, *args, **kwargs)
                    cls._cache[config_file] = instance

            return cls._cache[config_file]

        def _init_singleton(self, config_file: str, *args, **kwargs):
            # Call the original MaCh3TutorialWrapper initializer
            super().__init__(config_file)
            self._config_file = config_file  # store for reference

        def get_config_file(self):
            return self._config_file
else:
    class MaCh3TutorialInterface:
        def __init__(self, *args, kwargs):
            raise Exception("MACH3 TUTORIAL NOT SET UP CORRECTLY")

