Configuration Parsing
=========================

The ``XGCN.main.run_model`` module supports parsing model configurations 
from command line arguments and ``.yaml`` files. 
Directory ``config/`` includes ``.yaml`` configuration file templates for all the models, 
and directory ``scripts/`` provides ``.sh`` shell scripts to run the models. 

If you want to use a ``.yaml`` configuration file, specify the path 
with the command line argument ``--config_file`` like follows:

.. code:: bash

    python -m XGCN.main.run_model \
        --config_file "../config/GraphSAGE/config.yaml" \
        --seed 1999 \
        ...

Note that a ``.yaml`` file is not a necessity of running the code and has lower 
priority when the same command line argument is given. 
