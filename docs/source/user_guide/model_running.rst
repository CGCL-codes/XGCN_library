Model Training
=========================

Once the data preparation is done, users can easily run a model 
with the ``XGCN.main.run_model`` module: 

.. code:: bash

    python -m XGCN.main.run_model \
        --model "GraphSAGE" \
        --seed 1999 \
        --data_root ... \
        --results_root ... \
        ...

In this section, we introduce the model configuration and training results 
using the facebook dataset created in the previous subsection 
and the common model GraphSAGE as an example.

.. toctree::
    :maxdepth: 1

    model_running/config_parsing
    model_running/config_components
