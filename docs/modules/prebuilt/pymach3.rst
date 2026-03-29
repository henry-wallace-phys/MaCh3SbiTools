PyMaCh3
==========
For users of MaCh3 we provide a pre-built simulator for use with pyMaCh3-Tutorial.
It can be found
`here <https://github.com/henry-wallace-phys/MaCh3SbiTools/tree/main/src/mach3sbitools/examples/pyMaCh3>`_.

Installation instructions for pyMaCh3 can be found in the usual `MaCh3 repo <https://github.com/mach3-software/MaCh3/tree/develop>`_.
Currently the simulator is set up to work with `MaCh3 Tutorial <https://github.com/mach3-software/MaCh3Tutorial/tree/main>`_
and can be run with

.. code-block:: sh
   mach3sbi [simulate/create_prior/save_data] -m mach3sbitools.examples -c PyMaCh3 pyMaCh3Simulator [opts]
This is exactly the same as using the custom simulators discussed earlier.

This code provides an interface between MaCh3 and MaCh3SBITools as well as some helper functions to
read the necessary parts of the YAML configs used.

Working with your experiment Repo
------------------------------------
For most experiment's purposes, the only necessary change in the tutorial wrapper is to
swap out the instances of `SampleHandlerTutorial` with `SampleHandler<MyExperiment>`. It is
recommended to do this work in a separate directory following this structure

.. code-block:: text

    my_package
    ├── my_simulator
    └── __init__.py
Due to the relative pathing used in MaCh3 configurations it's often easiest to do this with the MaCh3-repo
itself.
