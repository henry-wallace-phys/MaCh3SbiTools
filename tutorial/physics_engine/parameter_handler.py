from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import multivariate_normal

from mach3sbitools.utils import get_logger

from .event_generator import EventModes

logger = get_logger("tutorial")


@dataclass
class Parameter:
    name: str
    is_flat: bool
    nominal: float
    error: float
    bounds: tuple[float, float]
    modes: list[int]
    energy_range: tuple[float, float]
    correlations: dict[str, float] = field(default_factory=dict)
    cos_weight: bool = False

    def __post_init__(self):
        # Just check we're nicely within bounds
        if self.bounds[0] > self.bounds[1]:
            raise ValueError(
                f"{self.name}: lower bound {self.bounds[0]} > upper bound {self.bounds[1]} for"
            )

        if self.nominal > self.bounds[1] or self.nominal < self.bounds[0]:
            raise ValueError(
                f"{self.name}: Nominal value {self.nominal} out of bounds [{self.bounds[0]}, {self.bounds[1]}]"
            )

        self.enum_mode = [EventModes(m) for m in self.modes]


class ParameterHandler:
    """
    A simple class to set/get parameter values
    """

    def __init__(self, config_file: Path | str, seed: int = 42):
        """
        :param config_file: Takes a config_file as input, returns a "parameter config:
        """
        if not isinstance(config_file, Path):
            config_file = Path(config_file)

        if not config_file.is_file():
            raise FileExistsError(
                f"{config_file} does not exist cannot make ParameterHandler"
            )

        with open(config_file) as file:
            config = yaml.safe_load(file)

        self.parameters = [Parameter(**p["Parameter"]) for p in config["Parameters"]]
        self.n_params = len(self.parameters)
        self.covariance_matrix = self._build_covariance_matrix()

        self.nominal_values = np.array(
            [self.get_nominal(i) for i in range(self.n_params)], dtype=float
        )

        self.parameter_values = np.array(
            [p.nominal for p in self.parameters], dtype=float
        )
        self.parameter_weights = self.parameter_values.copy()

        self._cos_mask = np.array([p.cos_weight for p in self.parameters])
        self._non_flat_idx = np.array(
            [not self.get_is_flat(i) for i in range(self.n_params)]
        )

        # Pre-compute bounds arrays for vectorised OOB check
        self._lower_bounds = np.array([p.bounds[0] for p in self.parameters])
        self._upper_bounds = np.array([p.bounds[1] for p in self.parameters])

        # For "physics"
        self.rng = np.random.default_rng(seed)
        self.distribution = multivariate_normal(
            mean=self.nominal_values[self._non_flat_idx],
            cov=self.covariance_matrix[self._non_flat_idx][:, self._non_flat_idx],
        )

        logger.info("Parameter handler initialised with %d parameters", self.n_params)

    # ------------------------------------------------
    # Param setter
    # ------------------------------------------------
    def set_parameter_values(self, new_values: np.ndarray):
        """
        Set your parameter values to some value
        :param new_values:
        :return:
        """
        if len(new_values) != self.n_params:
            raise ValueError(
                f"{len(new_values)} values are not equal to {self.n_params}"
            )
        np.copyto(self.parameter_values, new_values)  # in-place, no allocation

    def get_parameter_values(self):
        return self.parameter_values.view()  # zero-copy read-only view

    def get_parameter_weights(self):
        np.copyto(
            self.parameter_weights, self.parameter_values
        )  # one in-place copy into pre-allocated buffer
        self.parameter_weights[self._cos_mask] = (
            np.cos(self.parameter_weights[self._cos_mask] / 2) ** 2
        )
        return self.parameter_weights

    # ---------------------------------------------------------
    # "Physics" Utilities to make this handle a bit like MaCh3
    # ---------------------------------------------------------
    def throw_params(self, about: np.ndarray | None = None):
        """
        Throw parameter values about a point
        :param about: Place to throw about
        :return:
        """
        param_throw = self.rng.multivariate_normal(
            mean=np.zeros(self.n_params),
            cov=self.covariance_matrix,
        )

        if about is None:
            about = self.nominal_values
        else:
            if about.shape != self.parameter_values.shape:
                raise ValueError(
                    f"About shape {about.shape}  !=  Parameter shape {self.parameter_values.shape}"
                )

        self.set_parameter_values(param_throw + about)

    def get_log_prior(self, about: np.ndarray | None = None):
        """
        Gets the log prior. For out of bounds parameters returns -inf
        :param about:
        :return:
        """
        if about is None:
            about = self.parameter_values
        else:
            if about.shape != self.parameter_values.shape:
                raise ValueError(
                    f"About shape {about.shape}  !=  Parameter shape {self.parameter_values.shape}"
                )

        if self._get_oob(about):
            return -np.inf
        return -1 * self.distribution.logpdf(about[self._non_flat_idx])

    # ------------------------------------------------
    # Getters
    # [pls never have this many useless getters in real code!]
    # ------------------------------------------------
    def get_is_flat(self, param_index: int) -> bool:
        """
        Check if a given parameter is flat
        :param param_index: Index of the parameter
        :return: Whether or not the parameter is flat
        """
        return self.parameters[param_index].is_flat

    def get_nominal(self, param_index: int) -> float:
        """
        Get a given parameter's (prior) nominal value
        :param param_index:
        :return: Parameter nominal value
        """
        return self.parameters[param_index].nominal

    def get_error(self, param_index: int) -> float:
        """
        Get a given parameter's (prior) error
        :param param_index:
        :return: The error
        """
        return self.parameters[param_index].error

    def get_bounds(self, param_index: int) -> tuple[float, float]:
        """
        Get a given parameter's bounds
        :param param_index:
        :return: lower bound, upper bound
        """
        return self.parameters[param_index].bounds

    def get_name(self, param_index: int) -> str:
        """
        Get a given parameter's name
        :param param_index:
        :return: The parameter name
        """
        return self.parameters[param_index].name

    def get_n_params(self):
        return self.n_params

    # ---------------------------------------------
    # Internal Methods (Don't worry about these!)
    # ---------------------------------------------
    def _build_covariance_matrix(self):
        """
        Build the covariance matrix
        :return: np.ndarray
        """
        covariance_matrix = np.zeros((self.n_params, self.n_params))

        for i, par_i in enumerate(self.parameters):
            for j, par_j in enumerate(self.parameters[i:], start=i):
                if i == j:
                    covariance_matrix[i, j] = par_i.error**2
                else:
                    covariance_matrix[i, j] = covariance_matrix[j, i] = (
                        self._get_covariance(par_i, par_j)
                    )

        return covariance_matrix

    @classmethod
    def _get_covariance(cls, par_i: Parameter, par_j: Parameter):
        """
        Get the covariance between two parameters

        :param par_i: Parameter 1
        :param par_j: Parameter 2
        :return:
        """
        if par_j.name not in par_i.correlations:
            return 0

        if par_i.name not in par_j.correlations:
            raise ValueError(
                f"{par_i.name} correlated with {par_j.name} but this is not present in {par_j.name}"
            )

        if par_j.correlations[par_i.name] != par_i.correlations[par_j.name]:
            raise ValueError(
                f"{par_j.name} has correlation of {par_j.correlations[par_i.name]} with {par_i.name}"
                f"but {par_i.name} has correlation of {par_i.correlations[par_j.name]} with {par_j.name}"
            )

        return par_i.correlations[par_j.name] * par_i.error * par_j.error

    def _get_oob(self, about: np.ndarray) -> np.bool:
        # Short-circuits on first violated bound rather than checking all
        return np.any(about < self._lower_bounds) or np.any(about > self._upper_bounds)
