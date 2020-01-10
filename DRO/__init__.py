from __future__ import print_function, absolute_import

from .DRO_AVG import DRO_AVG
from .DRO_Grouping import DRO_Grouping
from .DRO_SAM import DRO_SAM
from .DRO_TOPK import DRO_TOPK
from .DRO import DRO


__factory = {
    'DRO_AVG': DRO_AVG,
    'DRO_Grouping': DRO_Grouping,
    'DRO_SAM': DRO_SAM,
    'DRO_TOPK': DRO_TOPK,
    'DRO': DRO
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
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)




