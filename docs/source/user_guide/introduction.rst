Introduction
================

The "User Guide" section is for those who want to quickly get started 
and run models on existing datasets or new datasets made by their own using XGCN. 

With XGCN's high-level APIs, users only need to focus on preparing the raw dataset input 
and setting model configurations, and they can easily run a model 
with the ``XGCN.main.run_model`` module: 

.. code:: bash

    python -m XGCN.main.run_model \
        --model "GraphSAGE" \
        --seed 1999 \
        --data_root ... \
        --results_root ... \
        ...
