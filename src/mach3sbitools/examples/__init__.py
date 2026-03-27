libs = []

try:
    from pyMaCh3 import pyMaCh3Simulator
except ImportError:
    pyMaCh3Simulator = None
else:
    libs.append("pyMaCh3Simulator")

__all__ = libs
