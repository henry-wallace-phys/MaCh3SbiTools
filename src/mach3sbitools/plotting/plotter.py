from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import pickle
from tqdm.autonotebook import tqdm

# For plotting!
class MaCh3SBIPlotter:
    def __init__(self, mach3_handler, posterior, n_samples: int=10_000_000, external_file: bool=False):
        self._handler = mach3_handler
        if external_file:
            with open(posterior, 'rb') as f:
                posterior = pickle.load(f)
        try:
            self._samples = posterior.sample((n_samples, )).cpu()
        except AttributeError:
            posterior.set_default_x(mach3_handler.get_data_bins()).train()
            self._samples = posterior.sample((n_samples, )).cpu()

        
        self._parameter_names = self._handler.get_parameter_names()

    @property
    def samples(self):
        return self._samples
    
    def plot(self, output_file: Path):
        with PdfPages(output_file) as pdf_file:
            for i in tqdm(range(len(self._samples[0])), desc="Making plots"):
                name = self._parameter_names[i]
                plt.hist(self._samples[:,i], bins=300, density=True)
                plt.xlabel(name)
                plt.ylabel("Posterior Density")
                pdf_file.savefig()
                plt.close()
