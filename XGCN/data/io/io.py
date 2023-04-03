import pickle as pkl
import json
import yaml
import gc


def save_pickle(filename, obj):
    """ Save objects using pickle. 
    
    Parameters
    ------------
    filename: str
        File path of the object.
    
    obj : the object to save
    
    Returns
    ---------
    None
    """
    with open(filename, "wb") as f:
        pkl.dump(obj, f)


def load_pickle(filename):
    """ load objects saved using pickle. 
    
    Parameters
    ------------
    filename : str
        File path of the object.
    
    Returns
    ---------
    The saved object.
    """
    with open(filename, "rb") as f:
        gc.disable()
        obj = pkl.load(f)
        gc.enable()
    return obj


def save_json(filename, obj):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(filename):
    with open(filename, 'r') as f:
        obj = json.load(f)
    return obj


def load_yaml(filename):
    with open(filename, 'r') as f:
        obj = yaml.load(f, Loader=yaml.FullLoader)
    return obj


def save_yaml(filename, obj):
    with open(filename, 'w') as f:
        yaml.dump(obj, f, indent=4, sort_keys=False)
