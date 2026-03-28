from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Systematic:
    name: str
    error: float
    nominal: float
    bounds: tuple[float, float]
    correlations: dict[str, float]
    flat_prior: bool
    fixed: bool


@dataclass
class ProcessedSystematics:
    names: np.ndarray
    errors: np.ndarray
    nominals: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    flat_priors: np.ndarray
    fixed: np.ndarray
    covariance: np.ndarray

    def __getitem__(self, mask):
        return ProcessedSystematics(
            names=self.names[mask],
            errors=self.errors[mask],
            nominals=self.nominals[mask],
            lower_bounds=self.lower_bounds[mask],
            upper_bounds=self.upper_bounds[mask],
            flat_priors=self.flat_priors[mask],
            fixed=self.fixed[mask],
            covariance=self.covariance[mask][:, mask],
        )

    def __len__(self):
        return len(self.names)


def _read_individual_yaml(
    parameter_yaml: Path, tune: str = "Generated"
) -> list[Systematic]:
    """
    Read a YAML file
    :param paramater_yaml:
    :return:
    """
    if not Path(parameter_yaml).is_file():
        raise FileExistsError(f"Cannot find file called {parameter_yaml}")

    with open(parameter_yaml) as f:
        yaml_data = yaml.safe_load(f)

    syst_arr = yaml_data.get("Systematics")

    if syst_arr is None:
        return []

    return [
        Systematic(
            name=s["Systematic"]["Names"]["FancyName"],
            error=s["Systematic"]["Error"],
            nominal=s["Systematic"]["ParameterValues"][tune],
            bounds=s["Systematic"]["ParameterBounds"],
            fixed=s["Systematic"].get("FixParam", False),
            flat_prior=s["Systematic"].get("FlatPrior", False),
            correlations={
                next(iter(d.keys())): next(iter(d.values()))
                for d in s["Systematic"].get("Correlations", [])
            },
        )
        for s in syst_arr
    ]


def process_parameter_yamls(
    parameter_yaml_list: list[Path],
    additional_fixed: list[str],
    tune: str = "Generated",
) -> ProcessedSystematics:
    """
    Process a list of MaCh3 parameter handler YAML files
    :param parameter_yaml_list:
    :return:
    """
    syst_arr = [y for p in parameter_yaml_list for y in _read_individual_yaml(p, tune)]

    n_systs = len(syst_arr)

    names = np.empty(n_systs, dtype=object)
    errors = np.empty(n_systs, dtype=float)
    nominals = np.empty(n_systs, dtype=float)
    lower_bounds = np.empty(n_systs, dtype=float)
    upper_bounds = np.empty(n_systs, dtype=float)
    flat_priors = np.empty(n_systs, dtype=bool)
    fix_pars = np.empty(n_systs, dtype=bool)

    # Gets a special treatment
    covariance_matrix = np.zeros((n_systs, n_systs), dtype=float)

    # Now flatten it
    for i, syst in enumerate(syst_arr):
        names[i] = syst.name
        lower_bounds[i] = syst.bounds[0]
        upper_bounds[i] = syst.bounds[1]
        errors[i] = syst.error
        nominals[i] = syst.nominal
        flat_priors[i] = syst.flat_prior
        fix_pars[i] = syst.fixed or syst.name in additional_fixed

        correlations = syst.correlations

        for j, other_syst in enumerate(syst_arr[i:], start=i):
            if i == j:
                covariance_matrix[i, j] = covariance_matrix[j, i] = syst.error**2
                continue

            if other_syst.name not in correlations:
                continue

            correlation = correlations[other_syst.name]

            covariance_matrix[i, j] = covariance_matrix[j, i] = (
                other_syst.error * syst.error * correlation
            )

    return ProcessedSystematics(
        names=names,
        errors=errors,
        nominals=nominals,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        flat_priors=flat_priors,
        fixed=fix_pars,
        covariance=covariance_matrix,
    )
