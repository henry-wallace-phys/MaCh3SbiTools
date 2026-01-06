from pathlib import Path

from mach3sbitools.sbi.sbi_factory import sbi_factory


found_mach3_dune = False
try:
    from mach3sbitools.mach3_interface.mach3_interface import MaCh3DUNEInterface
    found_mach3_dune = True

except Exception:
    ...

found_mach3_tutorial = False
try:
    from mach3sbitools.mach3_interface.mach3_interface import MaCh3TutorialInterface
    found_mach3_tutorial = True
    print("FOUND")
except Exception as e:
    print(e)
    
class MaCh3NotFoundError(Exception): ...

class MaCh3SbiUI:
    def __init__(self, input_config_path: Path, mach3_instance: str):
        print(f"Trying to open {input_config_path} with {mach3_instance}")
        
        if mach3_instance.lower() == "dune" and found_mach3_dune:
            try:
                print("Opening files with DUNE!")
                self._mach3 = MaCh3DUNEInterface(str(input_config_path))
            except Exception as e:
                raise MaCh3NotFoundError("Couldn't find MaCh3 DUNE!") from e
                
        elif mach3_instance.lower() == "tutorial" and found_mach3_tutorial:
            try:
                print("Opening files with Tutorial!")

                self._mach3 = MaCh3TutorialInterface(str(input_config_path))
            except Exception as e:
                raise MaCh3NotFoundError("Couldn't find MaCh3 Tutorial!") from e
        else:
            raise MaCh3NotFoundError(f"Couldn't find instance {mach3_instance}")
                

        self._fitter = None
        
    @property
    def mach3(self):
        return self._mach3
        
    def run_fit(self, fit_type, n_rounds, samples_per_round, sampling_settings={}, training_settings={}, autosave_interval: int=-1, output_file: Path = Path("model_output.pkl")):
        self._fitter = sbi_factory(fit_type, self._mach3, n_rounds, samples_per_round, autosave_interval=autosave_interval, output_file=output_file)
        self._fitter.train(sampling_settings, training_settings)
        
    @property
    def fitter(self):
        return self._fitter