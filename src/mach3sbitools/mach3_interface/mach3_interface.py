try:
    from pyMaCh3DUNE import MaCh3DUNEWrapper as MaCh3Wrapper
    print("Found MaCh3 DUNE instance, using that!")
except Exception:
    from pyMaCh3Tutorial import MaCh3TutorialWrapper as MaCh3Wrapper
    print("Found MaCh3 DUNE instance, using that!")
    
from threading import Lock

class MaCh3Interface(MaCh3Wrapper):
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