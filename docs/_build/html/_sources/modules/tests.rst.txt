Tests
=====

The test suite lives in ``tests/`` and uses pytest. A dummy simulator
(``tests/dummy_simulator/``) stands in for a real MaCh3 simulator, implementing
the full :class:`~mach3sbitools.simulator.simulator_injector.SimulatorProtocol` contract.

Dummy Simulator
---------------
.. autoclass:: dummy_simulator.DummySimulator
   :members:

.. autoclass:: dummy_simulator.PoorlyDefinedSimulator
   :members:

Test Configuration
------------------
.. automodule:: conftest
   :members: