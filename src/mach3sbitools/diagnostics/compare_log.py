from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.simulator import Simulator


def normalise_logl(input_arr: np.ndarray):
    mean = np.mean(input_arr)
    std_dev = np.std(input_arr)

    return (input_arr - mean) / std_dev


def compare_logl(
    simulator: Simulator,
    inference_handler: InferenceHandler,
    n_samples: int,
    save_path: Path | None = None,
):
    """Compares the LLH of an actual model and the simulator

    :param simulator: The simulator
    :param inference_handler: The inference handler
    :param n_samples: Number of samples to draw
    :param save_path: Where to save, defaults to None
    """

    simulator_samples, _ = simulator.simulate(n_samples)
    sample_llh = np.array(
        [
            simulator.simulator_wrapper.get_log_likelihood(t)
            for t in tqdm(simulator_samples, desc="Get LLH from simulator")
        ]
    )
    normalised_sample = normalise_logl(sample_llh)

    x_data = simulator.simulator_wrapper.get_data_bins()

    inference_llh = (
        inference_handler.get_log_likelihood(simulator_samples, x_data).cpu().numpy()
    )
    normalised_inf = normalise_logl(inference_llh)

    fig, (ax2d, ax1d) = plt.subplots(nrows=1, ncols=2)

    # 2D log-l/log-l plot
    log_l2d = ax2d.hist2d(
        x=normalised_sample, y=normalised_inf, density=True, cmap="hot"
    )
    fig.colorbar(log_l2d, ax=ax2d)
    ax2d.set_xlabel("Model Likelihood (Normalised)")
    ax2d.set_ylabel("SBI Likelihood (Normalised)")

    # Project to 1D
    min_val = np.min([normalised_inf, normalised_sample])
    max_val = np.max([normalised_inf, normalised_sample])
    bins = np.linspace(min_val, max_val, 100)

    ax1d.hist(normalised_sample, bins=bins, label="Sample Likelihood (Normalised)")
    ax1d.hist(normalised_inf, bins=bins, label="SBI Likelihood (Normalised)")
    ax1d.legend(loc="upper right")

    if plt.isinteractive():
        fig.show()

    if save_path is not None:
        fig.savefig(f"{save_path.stem}_2D.{save_path.suffix}")
