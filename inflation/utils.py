"""
This file contains auxiliary functions of general purpose
@authors: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens
"""
import numpy as np
from itertools import chain
from typing import Iterable


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
