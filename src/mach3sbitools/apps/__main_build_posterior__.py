from mach3sbitools.inference.sbi_interface import MaCh3SBIInterface

from pathlib import Path
import click
from tqdm import tqdm

from matplotlib import pyplot as plt
import fnmatch
from matplotlib.backends.backend_pdf import PdfPages

@click.command()
@click.option('--mach3-type', type=str, required=True, help='Type of MaCh3 simulator to use.')
@click.option('--config-file', type=Path, required=True, help='Path to the MaCh3 configuration file.')
@click.option('--mach3-dataset', type=Path, required=True, help='Path to the MaCh3 dataset folder.')
@click.option('--save-file', type=Path, required=True, help='Path to save the trained SBI inference file (pickle).')
@click.option('--nuisance-pars', type=str, multiple=True, help='List of nuisance variable substrings to exclude from the posterior.')
@click.option('--inference-file', type=Path, help='Path to the trained SBI inference file (pickle).')
# Posterior args
@click.option('--hidden-features', type=int, default=50, help='Number of hidden features in the neural network.')
@click.option('--num-transforms', type=int, default=20, help='Number of transforms in the neural network.')
@click.option('--dropout-probability', type=float, default=0.05, help='Dropout probability in the neural network.')
@click.option('--num-blocks', type=int, default=3, help='Number of blocks in the neural network.')

# Training args
@click.option('--batch-size', type=int, default=10_000, help='Batch size for training the SBI posterior.')
@click.option('--learning-rate', type=float, default=1e-4, help='Learning rate for training the SBI posterior.')
@click.option('--max-num-epochs', type=float, default=999999, help='Learning rate for training the SBI posterior.')

@click.option('--n-samples', type=int, default=1_000_000, help='Number of samples when generate thing plot.')

def main(mach3_type: str, config_file: Path, mach3_dataset: Path, nuisance_pars: list, inference_file: Path, save_file: Path,
         hidden_features: int, num_transforms: int, dropout_probability: float, num_blocks: int,
         batch_size: int, learning_rate: float, max_num_epochs: int, n_samples: int):
    # Load data
    inference = MaCh3SBIInterface(mach3_type, config_file, nuisance_pars)
    inference.set_dataset(mach3_dataset)

    
    if inference_file is None:
        inference.create_posterior(hidden_features=hidden_features, num_transforms=num_transforms, dropout_probability=dropout_probability, num_blocks=num_blocks)
    else:
        inference.load_inference(inference_file)
    
    for idx in range(len(inference.dataset)):
        inference.append_data(idx)

    print("Training")
    inference.train_posterior(
        save_file=save_file,
        training_batch_size=batch_size,
        learning_rate=learning_rate,
        max_num_epochs=max_num_epochs,
        resume_training=inference_file is not None,
    )
    
    s = inference.sample_posterior(n_samples)

    print(f"Plotting posterior to {save_file.stem}_posterior_plots.pdf")
    
    with PdfPages(f'{save_file.stem}_posterior_plots.pdf') as pdf:
        for i, name in enumerate(inference.simulator.mach3_wrapper.get_parameter_names()):
            # check if it's in nuisance vars
            if nuisance_pars is not None and any(fnmatch.fnmatch(name, n) for n in nuisance_pars):
                continue
            plt.figure()
            plt.hist(s[:, i].cpu().numpy(), bins=100, density=True)
            plt.title(f'Posterior of parameter {name}')
            plt.xlabel('Parameter value')
            plt.ylabel('Density')
            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    main()