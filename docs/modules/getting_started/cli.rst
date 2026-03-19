Running an Analysis
================================
Overview
---------------------------
The SBI analysis chain consists of 3 steps

    1. Run your simulations. This generates a large training set with which to work.

    2. Train the SBI model on these simulations

    3. Perform inference at a variety of data points and compare to the "real" simulator to ensure validity

MaCh3SBI tools provides a command line interface (CLI) with which to perform these steps. In addition it also has the following

* The ability to save your prior (taken from the simulator). This allows you avoid loading in your full simulator during training

* The ability to get "data" after processing from the simulator to ensure consistency

Using the CLI requires a :doc:`simulator to be setup <building_simulator>`.

The CLI
---------------------------
The main interface with MaCh3 SBI Tools is through the command line interface (CLI). For full CLI implementation details please see :doc:`the API documentation <../api_reference/cli>`.

The `mach3sbi` command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The entry point to the CLI is through `mach3sbi` invoked in the terminal. This is used to start the subcommands listed below. It can also display the subcommands through `--help`

..  code:: bash

        $ mach3sbi -h

        mach3sbi — simulation-based inference tools for MaCh3.

        Run ``mach3sbi COMMAND --help`` for detailed usage of each subcommand.

        Options:
          --log-level TEXT  Console logging level. One of DEBUG, INFO, WARNING, ERROR.
          --log_file TEXT   Optional path to write a plain-text log file.
          --help            Show this message and exit.

        Commands:
          create_prior  Generate and save a prior from a simulator instance.
          inference     Sample the posterior given observed data.
          save_data     Save observed data bins from the simulator to parquet.
          simulate      Run simulations and save to feather files.
          train         Train the NPE density estimator.

We will now discuss commands and the general Bayesian workflow!

Command: `mach3sbi simulate`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This command can be used to run many simulations from your simulator (see :doc:`here <building_simulator>` for information about setting up simulators). This can be invoked through

..  code:: bash

    mach3sbi simulate [-n number of simulations] -m [module path, i.e. my_package.my_simulator] -s [simulator class i.e. MySimulator] -c [simulator config i.e. FitterConf.yaml] -o [output file for sims, should end in .feather]

Additional options are discussed in the advanced set up. This code will run N simulations from your simulator and pipe them into a feather file. Best practise it to run many short instances of this job on a batch system to maximise the number of simulations.

Additionally, the `-r [path to prior]` flag can be passed. This will save the prior used for simulation into a file

Command: `mach3sbi create_prior`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In order to avoid having to load up the full simulator for doing SBI training, we can extract all the useful information from it into a prior file. This is invoked with
..  code:: bash

    mach3sbi create_prior -m [module i.e. mypackage.simulator] -s [simulator class i.e. MySimulator] -c [config i.e. config.yaml] -o [output file i.e. prior.pkl]

Command: `mach3sbi save_data`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Very often the data format/bin ordering used by simulator will change within the simulator. To ensure consistency we extract the data bins directly rather than using them as input. This will save the data to a paraquet file.
..  code:: bash

    mach3sbi save_data -m [module i.e. mypackage.simulator] -s [simulator class i.e. MySimulator] -c [config i.e. config.yaml] -o -o [output file i.e. data.paraquet]

.. note::
    Data here refers to the "observed" data poiny NOT your training data.

Command: `mach3sbi train`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This runs the full training loop. There are many options as detailed in :doc:`the API documentation <../api_reference/cli>`. This will cover the minimal useful set.

..  code:: bash

    mach3sbi save_data -s [file to save your model to] -d [the folder containing your training data]

There are a huge number of additional options for precisely tuning your model. The density estimator model types are those used in `SBI <https://sbi.readthedocs.io/en/stable/how_to_guide/03_density_estimators.html>`_ and can be set by string. Model dependent options are also discussed there.

In order to resume training simply set the `--resume_checkpoint` flag to point to the previous checkpoint.

Command: `mach3sbi inference`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This command allows you to perform inference with a completed simulator and saves to an Apache feather file. This can then be used for plotting. Currently plotting utilities are not provided but `Arviz <https://python.arviz.org/en/stable/>`_ provides good utilities.

..  code:: bash

    mach3sbi inference -i [path to your best model] -s [file to save inference to] -n [number of samples to do in the inference] -o [observed-data-file, produced by save_data]

Additionally if any of the model options (:doc:`see here <../api_reference/cli>`) were changed, these must also match i.e. the number of hidden layers + autoregressive transforms.
