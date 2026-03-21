import os
import sys

from mach3sbitools.utils import get_logger

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "physics_engine"))

logger = get_logger("tutorial")


class MySimulator:
    """
    Follow the instructions in Part 2: The SBI Simulator
    to build this class
    """
