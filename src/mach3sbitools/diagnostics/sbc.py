"""
SBC runs the simulation based calibration
"""

from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sbi.analysis.plot import plot_tarp, sbc_rank_plot
from sbi.diagnostics import check_tarp, run_sbc, run_tarp
from tqdm.auto import tqdm

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.simulator import Simulator
from mach3sbitools.utils import TorchDeviceHandler, get_logger

logger = get_logger()


class SBCDiagnostic:
    def __init__(
        self, simulator: Simulator, inference_handler: InferenceHandler, plot_dir: Path
    ):
        """
        Create the SBC diagnostics
        :param simulator: Simulator object
        :param inference_handler: InferenceHandler object
        :param plot_dir: Directory to save the SBC diagnostics
        """

        self.plot_dir = plot_dir
        self.plot_dir.mkdir(exist_ok=True, parents=True)

        self.simulator = simulator
        self._device_handler = TorchDeviceHandler()

        self.inference_handler = inference_handler
        inference_handler.build_posterior()
        self.posterior = inference_handler.posterior

        self.prior_samples: torch.Tensor | None = None
        self.prior_predictives: torch.Tensor | None = None

    def create_prior_samples(self, num_prior_samples: int) -> None:
        self.prior_samples = self.inference_handler.prior.sample((num_prior_samples,))

        prior_predictives_np = np.array(
            [
                self.simulator.simulator_wrapper.simulate(p)
                for p in tqdm(
                    self.prior_samples.cpu().numpy(), desc="Running SBC diagnostic"
                )
            ]
        )
        self.prior_predictives = self._device_handler.to_tensor(prior_predictives_np)

    def _check_prior_sampled(self):
        if self.prior_predictives is None:
            raise ValueError("Prior predictives not set")

        if self.posterior_predictives is None:
            raise ValueError("Posterior predictives not set")

    def rank_plot(
        self,
        num_posterior_samples: int = 1000,
        num_rank_bins: int = 20,
    ):
        """Run the SBC diagnostic
        Parameters
        :param: simulator - The simulator object
        :param: inference_handler - The inference_handler object
        :param: num_sbc_samples - The number of SBC samples
        :param: num_posterior_samples - The number of posterior samples
        """
        self._check_prior_sampled()

        ranks, _ = run_sbc(
            self.prior_samples,
            self.prior_predictives,
            self.posterior,
            num_posterior_samples=num_posterior_samples,
            use_batched_sampling=True,
        )

        fig, _ = sbc_rank_plot(
            ranks,
            num_posterior_samples,
            num_bins=num_rank_bins,
            figsize=(20, 20),
        )

        fig.savefig(self.plot_dir / "rank_plot.pdf")
        plt.close(fig)

    def expected_coverage(
        self,
        num_posterior_samples: int = 1000,
        num_rank_bins: int = 20,
    ):
        """Run the SBC diagnostic
        Parameters
        :param: simulator - The simulator object
        :param: inference_handler - The inference_handler object
        :param: num_sbc_samples - The number of SBC samples
        :param: num_posterior_samples - The number of posterior samples
        """
        self._check_prior_sampled()

        if self.posterior is None:
            raise ValueError("Posterior predictives not set")

        ranks, _ = run_sbc(
            self.prior_samples,
            self.prior_predictives,
            self.posterior,
            num_posterior_samples=num_posterior_samples,
            reduce_fns=lambda theta, x: -self.posterior.log_prob(theta, x),
            use_batched_sampling=True,
        )

        fig, _ = sbc_rank_plot(
            ranks,
            num_posterior_samples,
            num_bins=num_rank_bins,
            plt_type="cdf",
            figsize=(20, 20),
        )

        fig.savefig(self.plot_dir / "expected_coverage.pdf")
        plt.close(fig)

    def tarp(self, num_posterior_samples: int = 1000):
        self._check_prior_sampled()

        ecp, alpha = run_tarp(
            self.prior_samples,
            self.prior_predictives,
            self.posterior,
            references=None,
            num_posterior_samples=num_posterior_samples,
            use_batched_sampling=True,
        )

        atc, ks_pval = check_tarp(ecp, alpha)

        logger.info("ATC: {:4f}, should be close to 0", atc)
        logger.info("KS p-value: {:.4f}", ks_pval)

        fig, _ = plot_tarp(ecp, alpha)

        fig.savefig(self.plot_dir / "tarp.pdf")
        plt.close(fig)
