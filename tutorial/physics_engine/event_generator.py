from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from mach3sbitools.utils import get_logger

logger = get_logger("tutorial")


# -------------------------
# Event Spectra objects
# -------------------------
class EventModes(IntEnum):
    """Interaction mode flags for neutrino events."""

    CCMode = 0
    NCMode = 1


@dataclass
class EventSpectra:
    """Container for a collection of neutrino events with energy and mode labels.

    Bin indices are pre-computed on construction via :func:`numpy.digitize`.
    Events outside the bin range are assigned an internal index of ``-1`` and
    are excluded from all selection methods.

    :param energy_bins: Bin edges defining the energy histogram, shape ``(n_bins + 1,)``.
    :param energies: Reconstructed energy of each event, shape ``(n_events,)``.
    :param modes: Interaction mode of each event (see :class:`EventModes`), shape ``(n_events,)``.
    """

    energy_bins: np.ndarray
    energies: np.ndarray
    modes: np.ndarray

    def __post_init__(self):
        self.energy_bins = np.asarray(self.energy_bins, dtype=float)
        self.energies = np.asarray(self.energies, dtype=float)
        self.modes = np.asarray(self.modes, dtype=int)

        self.weights = np.ones(len(self.energies), dtype=float)

        n_bins = len(self.energy_bins) - 1
        raw_indices = np.digitize(self.energies, self.energy_bins) - 1
        self._bin_indices = np.where(
            (raw_indices >= 0) & (raw_indices < n_bins), raw_indices, -1
        )

    @property
    def binned_events(self) -> list[np.ndarray]:
        """All events grouped by energy bin, regardless of mode.

        :returns: One array of event indices per bin, ordered to match ``energy_bins``.
        :rtype: list[np.ndarray]
        """
        return [
            np.where(self._bin_indices == i)[0]
            for i in range(len(self.energy_bins) - 1)
        ]

    def reweight(self, weights: np.ndarray) -> None:
        """Replace the per-event weights with a new array.

        :param weights: New weights for every event, shape ``(n_events,)``.
            Must match the length of ``self.energies``.
        :type weights: np.ndarray
        :raises ValueError: If ``weights`` does not have the same length as
            ``self.energies``.
        """
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(self.energies):
            raise ValueError(
                f"weights length {len(weights)} does not match "
                f"number of events {len(self.energies)}"
            )
        self.weights = weights

    def reset_weights(self) -> None:
        """Reset all per-event weights to ``1.0``.

        Equivalent to calling :meth:`reweight` with an array of ones, but
        more explicit about intent.
        """
        self.weights = np.ones(len(self.energies), dtype=float)

    def apply_weight(
        self,
        weight: float,
        energy_low: float | None = None,
        energy_high: float | None = None,
        mode: list[EventModes] | None = None,
    ) -> None:
        mask = np.ones(len(self.energies), dtype=bool)
        if energy_low is not None:
            mask &= self.energies >= energy_low
        if energy_high is not None:
            mask &= self.energies < energy_high
        if mode is not None:
            mask &= np.isin(self.modes, mode)
        self.weights[mask] *= weight

    def get_weighted_hist(self) -> np.ndarray:
        n_bins = len(self.energy_bins) - 1
        return np.array(
            [self.weights[self._bin_indices == i].sum() for i in range(n_bins)],
            dtype=float,
        )


# Now we get extra Hacky
DEFAULT_SEED = 42


def generate_events(
    n: int, seed: int | None = DEFAULT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic sample of neutrino events.

    Energies are drawn from a Gamma distribution and clipped to ``[0, 10]``.
    Modes are assigned so that roughly 20% of events are NC and 80% are CC.

    :param n: Number of events to generate before energy clipping.
    :param seed: Seed for :class:`numpy.random.Generator`. Pass ``None`` for a
        non-deterministic result. Defaults to :data:`DEFAULT_SEED`.
    :returns: Tuple of ``(energies, modes)`` for surviving events in ``[0, 10]``.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    logger.debug("Generating {:d} events", n)

    rng = np.random.default_rng(seed)
    energies = rng.gamma(shape=3, scale=1, size=n)
    modes = rng.uniform(size=n)

    # Set modes to be 20% "NC" 80% CC
    modes[modes < 0.2] = EventModes.NCMode
    modes[modes >= 0.2] = EventModes.CCMode

    return energies, modes
