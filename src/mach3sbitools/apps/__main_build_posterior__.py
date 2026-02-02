from mach3sbitools.inference.sbi_interface import MaCh3SBIInterface

from pathlib import Path
import click

@click.command()
@click.option('--mach3-type', type=str, required=True, help='Type of MaCh3 simulator to use.')
@click.option('--config-file', type=Path, required=True, help='Path to the MaCh3 configuration file.')
@click.option('--mach3-dataset', type=Path, required=True, help='Path to the MaCh3 dataset folder.')
@click.option('--nuisance-vars', type=str, multiple=True, help='List of nuisance variable substrings to exclude from the posterior.')
@click.option('--inference-file', type=Path, required=True, help='Path to the trained SBI inference file (pickle).')
@click.option('--output-file', type=Path, required=True, help='Path to the output PDF file for posterior comparison.')
@click.option('--n-samples', type=int, default=1_000_000, help='Number of samples to draw from the posterior.')

# Posterior args
@click.option('--hidden-features', type=int, default=50, help='Number of hidden features in the neural network.')
@click.option('--num-transforms', type=int, default=20, help='Number of transforms in the neural network.')
@click.option('--dropout-probability', type=float, default=0.05, help='Dropout probability in the neural network.')
@click.option('--num-blocks', type=int, default=3, help='Number of blocks in the neural network.')

# Training args
@click.option('--batch-size', type=int, default=10_000, help='Batch size for training the SBI posterior.')
@click.option('--learning-rate', type=float, default=1e-4, help='Learning rate for training the SBI posterior.')
@click.option('--max_num_epochs', type=int, default=400, help='Maximum number of epochs for training the SBI posterior.')
@click.option('--lr_decay', type=float, default=0.1, help='Learning rate decay factor.')
@click.option("--min_lr", type=float, default=1e-8, help="Minimum learning rate.")

def main(mach3_type: str, config_file: Path, mach3_dataset: Path, nuisance_vars: list, inference_file: Path, output_file: Path, mach3_ttree: Path, n_samples: int, 
         hidden_features: int, num_transforms: int, dropout_probability: float, num_blocks: int,
         batch_size: int, learning_rate: float, max_num_epochs: int, lr_decay: float, min_lr: float):
    # Load data
    inference = MaCh3SBIInterface(mach3_type, config_file)
    inference.set_dataset(mach3_dataset)
    inference.create_posterior(hidden_features=hidden_features, num_transforms=num_transforms, dropout_probability=dropout_probability, num_blocks=num_blocks)
    for idx in range(len(inference.dataset)):
        inference.append_data(idx, nuisance_vars)
    
    inference.train_posterior(
        save_file=inference_file,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_num_epochs=max_num_epochs,
        lr_decay=lr_decay,
        min_lr=min_lr
    )
    
if __name__ == '__main__':
    main()