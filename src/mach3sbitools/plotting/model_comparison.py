from mach3sbitools.ui.sbi_ui import MaCh3SbiUI
import pickle
import uproot as ur
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tqdm.auto import tqdm

from mach3sbitools.utils.unpickler import CPU_Unpickler

def compare_models(ui: MaCh3SbiUI, output_file: str, input_chain: str, input_files: list[str], input_labels: list[str]):
    if len(input_files) != len(input_labels):
        raise ValueError("Number of input files must match number of input labels")
    
    print(f"Comparing models from {input_files} using chain {input_chain}. Saving results to {output_file}")
    
    samples = {}
    for file, label in zip(input_files, input_labels):
        with open(file, 'rb') as f:
            posterior = pickle.load(f)
            samples[label] = posterior.sample((1_000_000,), x=ui.mach3.get_data_bins())
            
    mach3_chain = ur.open(f"{input_chain}:posteriors").arrays(library="np")
    
    with PdfPages(output_file) as pdf:
        for i, name in tqdm(enumerate(ui.mach3.get_parameter_names())):
            _, bins, _ = plt.hist(mach3_chain[name][10000:], bins=50, density=True, alpha=0.5, label="MaCh3 Chain")

            for label, sample in samples.items():
                plt.hist(sample[:, i].cpu().numpy(), bins=bins, density=True, alpha=0.5, label=label)

            plt.legend()
            plt.xlabel(name)
            plt.ylabel("Density")
            plt.title(f"Comparison of posterior for {name}")
            pdf.savefig()
            plt.close()