from mach3sbitools.ui.sbi_ui import MaCh3SbiUI
from mach3sbitools.plotting.model_comparison import compare_models
from pathlib import Path

import click

@click.command()
@click.option('--mach3_config', '-c')
@click.option('--mach3_type', '-m', default='dune')
@click.option('--output_file', '-o', default="comparison_results.pdf")
@click.option('--input_chain', '-i')
@click.option('--input_files', '-f', multiple=True)
@click.option('--input_labels', '-l', multiple=True)
def main(mach3_config: str, mach3_type: str, output_file: str, input_chain: str, input_files: list[str], input_labels: list[str]):
    if len(input_files) != len(input_labels):
        raise ValueError("Number of input files must match number of input labels")
    
    ui = MaCh3SbiUI(Path(mach3_config), mach3_type)
    print(f"Comparing models from {input_files} using chain {input_chain}. Saving results to {output_file}")
    
    compare_models(ui, output_file, input_chain, input_files, input_labels)