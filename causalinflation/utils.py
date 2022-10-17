"""
This file contains auxiliary functions of general purpose
@authors: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens
"""
import numpy as np
from itertools import chain
from typing import Iterable


def blank_tqdm(*args, **kwargs):
    """Placeholder in case the tqdm library is not installed (see
    https://tqdm.github.io). If code is set to print a tqdm progress bar and
    tqdm is not installed, this just prints the ``desc`` argument.
    """
    try:
        if not kwargs["disable"]:
            print(kwargs["desc"])
    except KeyError:
        pass
    return args[0]


def flatten(nested):
    """Keeps flattening a nested lists of lists until  the
    first element of the resulting list is not a list.
    """
    if isinstance(nested, np.ndarray):
        return nested.ravel().tolist()
    else:
        while isinstance(nested[0], Iterable):
            nested = list(chain.from_iterable(nested))
        return nested
