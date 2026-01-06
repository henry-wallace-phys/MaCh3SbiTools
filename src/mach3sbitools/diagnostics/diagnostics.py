from typing import Optional

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
import numpy as np

import sbi.diagnostics as sbi_diag
from sbi.simulators.gaussian_mixture import (
    samples_true_posterior_gaussian_mixture_uniform_prior,
)


class SBIDiagnostics:
    def __init__(self, mach3_handler, posterior):
        self.mach3_handler = mach3_handler
        self.posterior = posterior
    
        self._lc2st_test = None
    
    
        self._thetas_star = []
        self._xs_star = []
        self._ref_samples_star = {}
        self._post_samples_star = []
        self._conf_alpha = 0.05
    
    
    def evaluate_posterior_estimator(self, n_samples: int=10000):

        # Avoid recomputing
        if self._thetas_star is not None:
            return
        
        self._thetas_star = self.mach3_handler.prior.sample((3, ))
        self._xs_star = self.mach3_handler.get_x_vals(self._thetas_star)
        self._ref_samples_star = {}
        for i,x in enumerate(self._xs_star):
            self._ref_samples_star[i] = samples_true_posterior_gaussian_mixture_uniform_prior(
                x,
                x_o=x[None,:]
            )
        self._post_samples_star = self.mach3_handler.posterior.sample_batched((n_samples), x=self._xs_star, max_sampling_batch_size=1000).permute(1,0,2)
    
    def calc_lc2st(self, n_samples: int=10000):
        if self._lc2st_test is not None:
            return self._lc2st_test
        
        # Sample from prior
        prior_theta = self.mach3_handler.prior.sample((n_samples, ))
        # Now simulate
        prior_x, _ = self.mach3_handler.get_x_vals(prior_theta)
        
        # Now sample the posterior
        posterior_x = self.mach3_handler.posterior.sample_batched((1,), x=prior_x)[0]
        
        # now set up the LC2ST
        self._lc2st_test = sbi_diag.LC2ST(
            thetas=prior_theta,
            xs = prior_x,
            posterior_samples=posterior_x,
            classifier="mlp",
            num_ensemble=5
        )
        _ = self._lc2st_test.train_under_null_hypothesis() # over 100 trials under (H0)
        _ = self._lc2st_test.train_on_observed_data() # on observed data
        
    def plot_full_lc2st(self, output_file: Path, conf_alpha: Optional[float]=None):
        if conf_alpha is not None:
            self._conf_alpha = conf_alpha
        
        self.calc_lc2st()
        self.evaluate_posterior_estimator()
        
        prob_score_dict = {}
        with PdfPages(output_file) as pdf_file:
            for p in tqdm(range(len(self._thetas_star)), desc="Making LC2ST plots"):
                self._make_lc2st_plot(p, pdf_file, prob_score_dict)
        
        # Print summary
        for param_name, vals in prob_score_dict.items():
            probs = vals['prob']
            scores = vals['score']
            print(f"Parameter: {param_name}")
            print(f"  Mean Probability: {np.mean(probs):.4f} | Std: {np.std(probs):.4f}")
            print(f"  Mean Score: {np.mean(scores):.4f} | Std: {np.std(scores):.4f}")
        
    def _make_lc2st_plot(self, p: int, pdf_file: PdfPages, prob_score_dict: dict={}):
        
        if self._lc2st_test is None:
            raise Exception("LC2ST test has not been calculated yet!")
        
        probs,scores = self._lc2st_test.get_scores(
            theta_o = self._post_samples_star[p],
            x_o = self._xs_star[p],
            return_probs=True,
            trained_clfs = self._lc2st_test.trained_clfs
        )
        
        prob_score_dict[self.mach3_handler.get_parameter_names()[p]] = {'prob': probs, 'score': scores}
        
        T_data = self._lc2st_test.get_statistic_on_observed_data(
            theta_o=self._post_samples_star[p],
            x_o=self._xs_star[p]
        )

        T_null = self._lc2st_test.get_statistics_under_null_hypothesis(
            theta_o=self._post_samples_star[p],
            x_o=self._xs_star[p]
        )

        p_value = self._lc2st_test.p_value(self._post_samples_star[p], self._xs_star[p])
        reject = self._lc2st_test.reject_test(self._post_samples_star[p], self._xs_star[p], alpha=self._conf_alpha)
        
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        quantiles = np.quantile(T_null, [0, 1-self._conf_alpha])
        ax.hist(T_null, bins=50, density=True, alpha=0.7, label="Null Hypothesis Distribution")
        ax.axvline(T_data, color='r', linestyle='--', label="Observed Statistic")
        ax.axvline(quantiles[0], color='k', linestyle='-.--', label=f"{(1-self._conf_alpha)*100}% Quantile")
        ax.axvline(quantiles[1], color='k', linestyle='-.--')

        ax.set_xlabel("LC2ST Statistic")
        ax.set_ylabel("Density")
        ax.set_xlim(min(quantiles[0], T_data)*0.9, max(quantiles[1], T_data)*1.1)
        ax.set_title(f"LC2ST Diagnostic for Theta Star {p}\nP-value: {p_value:.4f} | Reject H0 at alpha={self._conf_alpha}: {reject}")
        ax.legend()
        ax.set_title(self.mach3_handler.get_parameter_names()[p])
        pdf_file.savefig(fig)
        plt.close(fig)