from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from mach3sbitools.utils import get_logger

from .event_generator import EventSpectra, generate_events
from .parameter_handler import ParameterHandler

logger = get_logger("tutorial")


# -------------------------
# Actual Sample Handler
# -------------------------
@dataclass(frozen=True)
class Sample:
    bins: np.ndarray
    data: list[float] = field(default_factory=list)

    def __post_init__(self):
        if len(self.data) and len(self.data) != len(self.bins) - 1:
            raise ValueError("The number of bins and data do not match")


class SampleHandler:
    """
    A generic Sample Handling class. This performs reweights as well as letting the user obtain sample information
    """

    def __init__(
        self, sample_config_file: Path | str, parameter_handler: ParameterHandler
    ):
        if not isinstance(sample_config_file, Path):
            sample_config_file = Path(sample_config_file)

        with open(sample_config_file) as f:
            sample_loaded = yaml.safe_load(f)

        bin_info = sample_loaded["SampleSettings"]["BinningInfo"]
        bins = np.linspace(bin_info["low"], bin_info["high"], bin_info["n_bins"])

        self.samples = Sample(
            bins=bins, data=sample_loaded["SampleSettings"].get("data", [])
        )

        # Setup the parameter handler
        self.parameter_handler = parameter_handler

        # Will give us some nice data events, modes
        self._base_energy_spectra, self._base_energy_modes = generate_events(10_000)
        self.mc = EventSpectra(
            np.array(self.samples.bins),
            self._base_energy_spectra,
            self._base_energy_modes,
        )

        self.parameter_handler.set_parameter_values(
            self.parameter_handler.nominal_values
        )
        self.reweight()

        if not self.samples.data:
            # This is fine, we can get it from the spectra
            self.data = self.mc.get_weighted_hist()
        else:
            if (
                len(self.samples.data) != len(self.samples.bins) - 1
            ):  # fix: was comparing list to itself
                raise ValueError("The number of bins and data do not match")
            self.data = np.array(self.samples.data)

        self.n_bins = len(self.data)  # fix: set after self.data is finalised

        logger.info("Sample handler initialised from %s", sample_config_file)

    def reweight(self):
        self.mc.reset_weights()
        for par_val, par_info in zip(
            self.parameter_handler.get_parameter_weights(),
            self.parameter_handler.parameters,
        ):
            e_low, e_high = par_info.energy_range
            par_modes = par_info.enum_mode if par_info.enum_mode else None
            self.mc.apply_weight(par_val, e_low, e_high, par_modes)

    def get_likelihood(self):
        mc_vals = self.get_mc_vals()
        mc_safe = np.clip(mc_vals, 1e-10, None)
        return np.sum(self.data * (1 - np.log(self.data / mc_safe)) - mc_safe)

    def get_data(self):
        return self.data

    def get_mc_vals(self):
        return self.mc.get_weighted_hist()
