"""
This file contains auxiliary functions of general purpose

@authors: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens
"""
from __future__ import print_function
import sympy

import numpy as np
import scipy.sparse as sps

from itertools import chain
from typing import Any, Dict, Iterable, List, Tuple, Union
from sys import stderr
from operator import itemgetter
from collections import deque


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
    if not cert:
        return cert
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


def partsextractor(thing_to_take_parts_of, indices) -> Tuple[int,...]:
    if hasattr(indices, '__iter__'):
        if len(indices) == 0:
            return tuple()
        elif len(indices) == 1:
            return (itemgetter(*indices)(thing_to_take_parts_of),)
        else:
            return itemgetter(*indices)(thing_to_take_parts_of)
    else:
        return itemgetter(indices)(thing_to_take_parts_of)


def expand_sparse_vec(sparse_vec: sps.coo_array,
                      conversion_style: str = "eq") -> sps.coo_array:
    """Expand a one-dimensional sparse matrix to its full form. Used to expand
    the solver arguments known_vars, lower_bounds, and upper_bounds."""
    assert conversion_style in {"eq", "lb", "ub"}, \
        "Conversion style must be `lb`, `ub`, or `eq`."
    nof_rows = sparse_vec.nnz
    nof_cols = sparse_vec.shape[1]
    if conversion_style == "eq":
        # Data values do not appear in '1' monomial column
        row = np.arange(nof_rows)
        col = sparse_vec.col
        data = np.ones(nof_rows)
    else:
        # Data values appear in '1' monomial column
        # Upper bound format: x <= a -> a - x >= 0
        row = np.repeat(np.arange(nof_rows), 2)
        col = np.vstack((
            sparse_vec.col,
            np.zeros(nof_rows)  # Assumes '1' monomial is first column
        )).T.ravel()
        data = np.vstack((
            -np.ones(nof_rows),
            sparse_vec.data,
        )).T.ravel()
    if conversion_style == "lb":
        # Lower bound format: x >= a -> x - a >= 0
        data = -data
    return sps.coo_array((data, (row, col)), shape=(nof_rows, nof_cols))


def vstack(blocks: tuple, format: str = 'coo') -> sps.coo_array:
    """Stack sparse matrices in coo_array form more efficiently."""
    non_empty = tuple(mat for mat in blocks if mat.shape[0])
    nof_blocks = len(non_empty)
    if nof_blocks > 1:
        if all(isinstance(block, sps.coo_array) for block in blocks):
            nof_rows = 0
            nof_cols = 0
            adjusted_rows = []
            for block in blocks:
                adjusted_rows.append(block.row + nof_rows)
                (block_len, block_wid) = block.shape
                nof_rows += block_len
                nof_cols = max(nof_cols, block_wid)
            mat_row = np.hstack(adjusted_rows)
            mat_col = np.hstack(tuple(block.col for block in blocks))
            mat_data = np.hstack(tuple(block.data for block in blocks))
            return sps.coo_array((mat_data, (mat_row, mat_col)),
                                  shape=(nof_rows, nof_cols)).asformat(format)
        else:
            return sps.vstack(blocks, format)
    elif nof_blocks == 1:
        return non_empty[0]
    else:
        return sps.coo_array([])


def perm_combiner(old_perms: np.ndarray, new_perms: np.ndarray) -> np.ndarray:
    combined = np.take(old_perms, new_perms, axis=1)
    new_shape = (-1, new_perms.shape[1])
    return combined.reshape(new_shape)


def all_and_maximal_cliques(adjmat: np.ndarray,
                            max_n=0,
                            isolate_maximal=True) -> (List, List):
    """Based on NetworkX's `enumerate_all_cliques`.
    This version uses native Python sets instead of numpy arrays.
    (Performance comparison needed.)

    Parameters
    ----------
    adjmat : numpy.ndarray
      A boolean numpy array representing the adjacency matrix of an undirected graph.
    max_n : int, optional
      A cutoff for clique size reporting. Default 0, meaning no cutoff.
    isolate_maximal : bool, optional
      A flag to disable filtering for maximality, which can increase performance. True by default.

    Returns
    -------
    Tuple[List, List]
      A list of all cliques as well as a list of maximal cliques. The maximal cliques list will be empty if the
      `isolate_maximal` flag is set to False.
    """
    all_cliques = [[]]
    maximal_cliques = []
    verts = tuple(range(adjmat.shape[0]))
    initial_cliques = [[u] for u in verts]
    nbrs_mat = np.triu(adjmat, k=1)
    initial_cnbrs = [np.flatnonzero(nbrs_mat[u]).tolist() for u in verts]
    nbrs = list(map(set, initial_cnbrs))
    queue = deque(zip(initial_cliques, initial_cnbrs))
    there_is_a_cutoff = (max_n <= 0)
    while queue:
        base, cnbrs = queue.popleft()
        all_cliques.append(base)
        if isolate_maximal and not len(cnbrs):
            base_as_set = set(base)
            if not any(base_as_set.issubset(superbase) for (superbase, _) in queue):
                maximal_cliques.append(base)
        elif there_is_a_cutoff or len(base) < max_n:
            for i, u in enumerate(cnbrs):
                new_base = base.copy()
                new_base.append(u)
                new_cnbrs = list(filter(nbrs[u].__contains__, cnbrs[i+1:]))
                queue.append((new_base, new_cnbrs))
    return all_cliques, maximal_cliques


def make_numerical(symbolic_expressions: Dict[Any, sympy.core.expr.Expr],
                   symbols_to_values: Dict[sympy.core.symbol.Symbol, float]
                   ) -> Dict[Any, float]:
    """Replace the symbols in the values of a dictionary by the corresponding
    numerical values.
    Parameters
    ----------
    symbolic_expressions : Dict[Any, sympy.core.expr.Expr]
        Dictionary where the values are symbolic expressions of some variables.
    symbols_to_values : Dict[sympy.core.symbol.Symbol, float]
        Correspondence of the variables in the expressions and their associated
        numerical values.
    Returns
    -------
    Dict[Any, float]
        The dictionary with same keys and evaluated expressions as values.
    """
    numeric_values = dict()
    for k, v in symbolic_expressions.items():
        try:
            numeric_values[k] = float(v.evalf(subs=symbols_to_values))
        except AttributeError:
            numeric_values[k] = float(v)
    return numeric_values
