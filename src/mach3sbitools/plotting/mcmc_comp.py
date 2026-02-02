from pathlib import Path
import uproot as ur
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from mach3sbitools.inference.sbi_interface import MaCh3SBIInterface

class MaCh3MCMCComparator:
    def __init__(self, mach3_name: str, config_file: Path):
        self.sbi_interface = MaCh3SBIInterface(mach3_name, config_file)

    def compare_posteriors(self, outfile: Path, inference_file: Path, mach3_ttree: Path, n_samples: int = 1_000_000) -> None:
        self.sbi_interface.load_inference(inference_file)
        self.sbi_interface.build_posterior()
        
        posterior_samples = self.sbi_interface.sample_posterior(n_samples)
        
        with ur.open(mach3_ttree) as file:
            tree = file["posteriors"].arrays(library="np")
        
        self.plot(posterior_samples, tree, outfile)
            
    def plot(self, ml, tree, output):
        with PdfPages(output) as pdf:
            for i, name in enumerate(self.sbi_interface.simulator.get_parameter_names()):
                ml_sample = ml[:, i]
                tree_sample = tree[name][200000:]
            
                if name == "delm2_23":
                    _, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
            
                    for ax, cut_label, cut_fn in [
                        (axes[0], "IO", lambda x: x < 0),
                        (axes[1], "NO", lambda x: x > 0),
                    ]:
                        # apply cuts independently
                        ml_cut = ml_sample[cut_fn(ml_sample)]
                        # saved_cut = saved_sample[cut_fn(saved_sample)]
                        tree_cut = tree_sample[cut_fn(tree_sample)]
            
                        # bins defined from ML sample only
                        _, bins = np.histogram(ml_cut.cpu(), bins=100, density=True)
            
                        ax.hist(tree_cut, bins=bins, density=True,
                                histtype="step", linewidth=2, label="MCMC", color="k")
                        ax.hist(ml_cut.cpu(), bins=bins, density=True,
                                histtype="step", linewidth=2, label="ML")
                        ax.set_xlabel(f"{name} ({cut_label})")
                        ax.set_ylabel("Density")
                        ax.legend(loc='upper left', fontsize='x-small')
            
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                    continue
            
                # ---- default path ----
                _, bins = np.histogram(ml_sample.cpu(), bins=100, density=True)
            
                plt.hist(tree_sample, bins=bins, density=True,
                        histtype="step", linewidth=2, label="MCMC", color="k")
                plt.hist(ml_sample.cpu(), bins=bins, density=True,
                        histtype="step", linewidth=2, label="ML")
                
                plt.legend(loc='upper left', fontsize='x-small')
                plt.xlabel(name)
                plt.ylabel("Density")
                pdf.savefig()
