Diagnostics Reference
#########################

We provide access to several diagnostics. The in-house "log_l" diagnostic is an adhoc
comparison of your model's LLH to the actual LLH.

Currently all other diagnostics are provided by the `SBI package <https://sbi.readthedocs.io/en/stable/how_to_guide/diagnostics.html>`_

Diagnostics
-------------

.. autoclass:: mach3sbitools.diagnostics.SBCDiagnostic
   :members:

.. autofunction:: mach3sbitools.diagnostics.compare_logl()
