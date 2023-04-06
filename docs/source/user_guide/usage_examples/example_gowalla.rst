Dara Processing Example: gowalla
=====================================

Here we present a user-item graph example. The used gowalla dataset can be found 
in our XGCN repository: ``data/raw_gowalla/``, which is copied from LightGCN's official code repository: 
https://github.com/gusye1234/LightGCN-PyTorch.

We recommend to arrange the data with a clear directory structure. 
To get started, you may manually setup a ``XGCN_data`` directory as follows: 
(It's recommended to put your ``XGCN_data`` somewhere else than in this repository.)

.. code:: 

    XGCN_data
    └── dataset
        └── raw_gowalla
            ├── train.txt
            └── test.txt

If you like to do the data process in JupyterNotebook, please refer to 
``script/data_process/gowalla/dataset_instance_making.ipynb``.

Firstly, we import some modules: 

.. code:: python

    >>> import dgl
    >>> import XGCN
    >>> from XGCN.data import io
    >>> from XGCN.utils.utils import ensure_dir, set_random_seed
    >>> import os.path as osp

Specify the data root and the dataset name: 

.. code:: python

    >>> all_data_root = '../../XGCN_data'  # write your own data root here
    >>> dataset = 'gowalla'

Load the ``train.txt`` as numpy array, and convert it to DGLGraph:

.. code:: python

    >>> raw_data_root = osp.join(all_data_root, 'dataset/raw_' + dataset)
    >>> E_src, E_dst = io.load_txt_adj_as_edges(osp.join(raw_data_root, 'train.txt'))
    >>> 
    >>> num_users = E_src.max() + 1
    >>> num_items = E_dst.max() + 1
    >>>
    >>> E_dst += 1  # range of item IDs in the DGLGraph: [num_users, num_users + num_items)
    >>>
    >>> g = dgl.graph((E_src, E_dst))

The gowalla dataset already has an evaluation set, so we just convert it to 
the XGCN's format:

.. code:: python

    >>> test_set = io.from_txt_adj_to_adj_eval_set(osp.join(raw_data_root, 'test.txt'))

By add an ``info`` Dict, we have already generated a complete dataset instance:

.. code:: python

    >>> info = {'graph_type': 'user-item', 'num_nodes': g.num_nodes(), 'num_edges': g.num_edges()}
    >>> 
    >>> data_root = osp.join(all_data_root, 'dataset/instance_' + dataset)
    >>> ensure_dir(data_root)  # make the directory if it doesn't exist
    >>> io.save_yaml(osp.join(data_root, 'info.yaml'), info)
    >>> io.save_pickle(osp.join(data_root, 'g.pkl'), g)
    >>> io.save_pickle(osp.join(data_root, 'test_set.pkl'), test_set)

If you have done the above steps successfully, your data directory will be like follows: 

.. code:: 

    XGCN_data
    └── dataset
        ├── raw_gowalla
        |   ├── train.txt
        |   └── test.txt
        └── instance_gowalla
            ├── info.yaml
            ├── g.pkl
            └── test_set.pkl
