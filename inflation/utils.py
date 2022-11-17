"""
This file contains auxiliary functions of general purpose
@authors: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens
"""
import numpy as np
from itertools import chain
from typing import Iterable, Union, List


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

def format_permutations(array: Union[np.ndarray, List[int]]) -> np.ndarray:
    """Permutations of inflation indices must leave the integers 0,
    corresponding to sources not being measured by the operator, invariant.
    In order to achieve this, this function shifts a permutation of sources
    by 1 and prepends it with the integer 0.

    Parameters
    ----------
    array : numpy.ndarray
        2-d array where each row is a permutations.

    Returns
    -------
    numpy.ndarray
        The processed list of permutations.
    """
    source_permutation = np.asarray(array) + 1
    padding = np.zeros((len(source_permutation), 1), dtype=int)
    return np.hstack((padding, source_permutation))
