================================
Installation
================================
Getting the Repo
--------------------------------
This repo currently does not have a PyPy distribution. As a result it must be cloned directory

.. code-block:: bash

    # Clone from github
    git clone https://github.com/mach3-software/MaCh3SbiTools/tree/main MaCh3SbiTools
    cd MaCh3SbiTools

Installing with Pip
--------------------------------
The install process with pip is as follows:

To install:

.. code-block:: bash

    # Ensure you have virtual env available
    pip install virtualenv

    # Create a virtual environment
    virtualenv .venv

    # Now source the venv. This is necessary whenever you want to use this package
    source .venv/bin/activate

    # Install via pip
    pip install .

.. note::
    Whilst UV is supported by this package, `MaCh3`, a common simulator/fitter used with MaCh3SBI tools does NOT support UV.

.. note::
    Conda may also be used but this is not tested/documented.
