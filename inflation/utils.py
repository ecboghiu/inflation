"""
This file contains auxiliary functions of general purpose
@authors: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens
"""
from __future__ import print_function
import numpy as np
from itertools import chain
from typing import Iterable, Union, List, Tuple, Dict
from sys import stderr
from operator import itemgetter
from scipy.sparse import coo_matrix


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

def format_permutations(array: Union[
    np.ndarray,
    List[List[int]],
    List[Tuple[int,...]],
    Tuple[Tuple[int,...],...],
    Tuple[List[int],...],
]) -> np.ndarray:
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
    return np.pad(source_permutation, ((0, 0), (1, 0)))

def clean_coefficients(cert: Dict[str, float],
                       chop_tol: float = 1e-10,
                       round_decimals: int = 3) -> Dict:
    """Clean the list of coefficients in a certificate.

    Parameters
    ----------
    cert : Dict[str, float]
      A dictionary containing as keys the monomials associated to the elements
      of the certificate and as values the corresponding coefficients.
    chop_tol : float, optional
      Coefficients in the dual certificate smaller in absolute value are
      set to zero. Defaults to ``1e-10``.
    round_decimals : int, optional
      Coefficients that are not set to zero are rounded to the number
      of decimals specified. Defaults to ``3``.

    Returns
    -------
    np.ndarray
      The cleaned-up coefficients.
    """
    chop_tol = np.abs(chop_tol)
    coeffs = np.asarray(list(cert.values()))
    if chop_tol > 0:
        # Try to take the smallest nonzero one and make it 1, when possible
        normalising_factor = np.min(np.abs(coeffs[np.abs(coeffs) > chop_tol]))
    else:
        # Take the largest nonzero one and make it 1
        normalising_factor = np.max(np.abs(coeffs[np.abs(coeffs) > chop_tol]))
    coeffs /= normalising_factor
    # Set to zero very small coefficients
    coeffs[np.abs(coeffs) <= chop_tol] = 0
    # Round
    coeffs = np.round(coeffs, decimals=round_decimals)
    return dict(zip(cert.keys(), coeffs.flat))

def eprint(*args, **kwargs):
    print(*args, file=stderr, **kwargs)

def partsextractor(thing_to_take_parts_of, indices):
    if hasattr(indices, '__iter__'):
        if len(indices) == 0:
            return tuple()
        elif len(indices) == 1:
            return (itemgetter(*indices)(thing_to_take_parts_of),)
        else:
            return itemgetter(*indices)(thing_to_take_parts_of)
    else:
        return itemgetter(indices)(thing_to_take_parts_of)


def sparse_vec_to_sparse_mat(sparse_vec: coo_matrix,
                             row_repeat: int = 1) -> coo_matrix:
    """Convert a one-dimensional sparse matrix to a full-dimensional one. Used
    to expand solver arguments that are passed as a one-dimensional matrix such
    as known_vars, lower_bounds, upper_bounds."""
    nof_rows = sparse_vec.nnz
    nof_cols = sparse_vec.shape[1]
    row = [*range(nof_rows)]
    col = sparse_vec.col
    data = [1] * nof_rows
    # Formulating variable bounds as inequality constraints in InflationLP
    if row_repeat == 2:
        row = np.repeat(np.arange(nof_rows), 2)
        col = [None] * (2 * nof_rows)
        col[::2] = sparse_vec.col
        col[1::2] = np.zeros(nof_rows)  # Assumes '1' monomial is first column
        data = [None] * (2 * nof_rows)
        data[::2] = [1] * nof_rows
        data[1::2] = [-c for c in sparse_vec.data]
    return coo_matrix((data, (row, col)), shape=(nof_rows, nof_cols))
