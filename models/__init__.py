from .BN_Inception import BN_Inception
from .DeepID2 import DeepID2
__factory = {
    'BN-Inception': BN_Inception,
    'DeepID2': DeepID2
}

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)
