===========================
Building Your simulator
===========================

What is a Simulator?
------------------------
A simulator is simply a piece of software that takes physics parameters as inputs like :math:`\delta_{CP}`, :math:`\Delta m^{2}_{32}` :cite:`the_mach3_collaboration_2026_18627288` or the many systematic parameters that make up modern physics models. It then outputs some observable, most commonly some kind of energy spectrum.

MaCh3 as a simulator
------------------------
`MaCh3 <https://github.com/mach3-software/MaCh3/tree/develop>`_ is a piece of software used across Neutrino Oscillation experiments used for Bayesian inference. As part of its Posterior likelihood calculation, it generates a set "expected" neutrino event spectra and compares it to some fixed data. For more information see the `MaCh3 Wiki <https://mach3-software.github.io/MaCh3/BayesianAnalysis.html>`_.

We can force it to work as a simulator by simply extracting these "expected spectra" at this point and sampling across a large number of points. There is one key caveat to this; neutrino events are independent and identically distributed random variables. As a result, we can model them with a Poisson distribution. When comparing to data in MaCh3 this is fine since we account for this within the likelihood function. For a true simulator though, we also need to simulate this randomness! This is fixed by simply applying Poisson fluctuations to the MaCh3 spectra!

Defining A Simulator
------------------------
MaCh3 SBI Tools expects a very particular Simulator format as defined in :py:class:`mach3sbitools.simulator.simulator_injector.SimulatorProtocol`.

Your "proper" simulator should be written in Python/have python bindings. The interface should then follow this skeleton:

.. code-block:: python

    from mach3sbitools.simulator.SimulatorProtocol

    # Also need your "proper" simulator
    from proper_simulator import ProperSimulator

    class MySimulator(SimulatorProtocol):
        def __init__(self, config_file: str):
            # Initialise the actual simulator
            self._proper_simulator = ProperSimulator(config_file)

        def simulate(self, theta: list[float])->list[float]:
            # Run the actual simulation for example:
            self._proper_simulator.set_values(theta)
            self._proper_simulator.reweight()
            return self._proper_simulator.get_mc()

        def get_parameter_names(self)->list[str]:
            # Get the parameter names, for example:
            return self._proper_simulator.parameters.get_names()

        def get_parameter_bounds(self)->list[float], list[float]
            # Get the upper/lower bounds for example:
            lower_bnd = self._proper_simulator.parameters.lower_bounds()
            upper_bnd = self._proper_simulator.parameters.upper_bounds()

            return lower_bnd, upper_bnd

        def get_is_flat(self, int i)->bool:
            # Holdover from MaCh3 where everything is either flat or Gaussian prior
            # Checks if a given input has a flat prior
            return self._proper_simulator.parameters.is_flat(i)

        def get_data_bins(self)->list[float]:
            # Get the actual bin heights for data
            return = []
            for s in self._proper_simulator.samples:
                return.extend(s.get_bins()[0])

        def get_parameter_nominals(self)->list[float]:
            # Get the prior nominal values for each parameter
            return self._proper_simulator.parameters.nominals()

        def get_parameter_errors(self)->list[float]:
            # Get the prior uncertainties for each parameter
            return self._proper_simulator.parameters.errors()

        def get_log_likelihood(self, theta):
            # Get the model likelihood!
            return self._proper_simulator.get_loglikelihood()

        def get_covariance_matrix(self):
            # Get the prior covariance matrix for all parameters
            return self._proper_simulator.parameters.get_cov()

Once this simulator is defined it should be stored in a small python package, for example

.. code-block:: text

    my_package
    ├── my_simulator
    └── __init__.py

It can then be used for priors/simulation with `Module=my_package.my_simulator` and `SimulatorClass=MySimulator`. For more information about this please see the :doc:`CLI guide <cli>`

.. note::
    It is not strictly necessary to inherit from `SimulatorProtocol` this will just help the linter/your IDE check for any non-implemented methods.
