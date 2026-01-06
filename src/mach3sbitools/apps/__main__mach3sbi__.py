from mach3sbitools.ui.sbi_ui import MaCh3SbiUI
from mach3sbitools.plotting.plotter import MaCh3SBIPlotter
from pathlib import Path

import click

@click.command()
@click.option('--input_file', '-i')
@click.option('--mach3_type', '-m', default='dune')
@click.option('--output_file', '-o', default="fitter.pkl")
@click.option('--fit_type', '-f', default='truncated_proposal')
@click.option('--autosave_interval', '-a', default=-1)
@click.option('--n_rounds', '-n', default=100)
@click.option('--samples_per_round', '-s', default=100)
def main(input_file: str, mach3_type: str, fit_type: str, output_file: str, autosave_interval: int, n_rounds: int, samples_per_round: int):
    ui = MaCh3SbiUI(Path(input_file), mach3_type)
    print(f"Running fit from {input_file}. Saving every {autosave_interval} steps. Running {n_rounds} rounds with {samples_per_round} samples per round")
    
    ui.run_fit(fit_type,
               n_rounds=n_rounds,
               samples_per_round=samples_per_round,
               sampling_settings={},
               training_settings={},
               autosave_interval=autosave_interval,
               output_file=Path(output_file))
    
    ui.fitter.save(Path(output_file))
    
    plotter = MaCh3SBIPlotter(ui.mach3, ui.fitter.posterior, 10_000_000, False)
    plotter.plot(Path(output_file).with_suffix('.pdf'))
    
    ui.fitter.get_sbc(out_folder=Path(output_file).parent, n_simulations=10000, num_posterior_samples=10000)