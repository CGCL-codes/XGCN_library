Overview
======================

This "USER GUIDE" section is for those who want to quickly get started 
and run the models on existing datasets or new datasets made by their own. 

The users only need to focus on preparing the raw dataset input 
(see the "Data Preparation" subsection) and 
and setting model configurations (see the "Model Training" subsection). 

We provide APIs such as ``XGCN.data.edges_split`` to split data for link prediction task. 
And users can run a model through the ``XGCN.data.edges_split`` module like follows:

.. code:: bash

    python -m XGCN.main.run_model \
        --model "xGCN" \
        --seed 1999 \
        --data_root ... \
        --results_root ... \
        ...

The results (including training log, model performance and time consumption) 
will be automatically saved in the specified directory.
