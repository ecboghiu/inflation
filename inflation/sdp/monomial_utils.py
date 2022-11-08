"""
This file contains auxiliary functions for the Monomial classes defined in
monomial_classes.py.

@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np
import sympy

from collections import Counter
from typing import List


def compute_marginal(prob_array: np.ndarray, atom: np.ndarray) -> float:
    """Given an atomic monomial and a probability distribution ``prob_array``,
    called as ``prob_array[a,b,c,...,x,y,z,...]``, returns the value of the
    probability associated to the monomial. (A numeric value if the distribution
    is numeric, or a symbolic expression if the distribution in an array of
    SymPy expressions.) It is possible to insert atomic monomials corresponding
    to marginals (i.e., where the list of first elements does not contain all
    the parties in the scenario) and the function will automatically compute all
    the summations over ``p[a,b,c,...,x,y,z,...]``.

    Parameters
    ----------
    prob_array : numpy.ndarray
        The target probability distribution. The dimensions of this array are
        (*outcomes_per_party, *settings_per_party), where the settings are
        explicitly 1 is there is only one measurement performed.
    atom : np.ndarray
        Monomial indicating a (commuting) collection of measurement operators.
        The array is a list of length-3 vectors. The first element indicates
        the party, the second element indicates the setting, and the final
        element indicates the outcome.

    Returns
    -------
    float
        The value of the symbolic probability (which can be a marginal)
    """
    if len(atom):
        n_parties: int = prob_array.ndim // 2
        # Return to pythonic numbering of parties
        participating_parties = atom[:, 0] - 1
        inputs  = atom[:, -2].astype(int)
        outputs = atom[:, -1].astype(int)
        indices_to_sum = list(set(range(n_parties)
                                  ).difference(participating_parties))
        marginal_dist  = np.sum(prob_array, axis=tuple(indices_to_sum))
        input_list: np.ndarray = np.zeros(n_parties, dtype=int)
        input_list[participating_parties] = inputs
        outputs_inputs = np.concatenate((outputs, input_list))
        return marginal_dist[tuple(outputs_inputs)]
    else:
        return 1.


def name_from_atom_names(atom_names: List[str]) -> str:
    """Join the names of monomials to a single string.

    Parameters
    ----------
    atom_names : List[str]
        The list of names of the atoms.

    Returns
    -------
    str
        The string representing the name of the product of the monomials.
    """
    if len(atom_names):
        output = ""
        for i, (name, power) in enumerate(Counter(atom_names).items()):
            if i > 0:
                output += "*"
            output += name
            if power > 1:
                output += "^" + str(power)
        return output
    else:
        return "1"


def symbol_from_atom_name(atomic_name: str) -> sympy.core.symbol.Symbol:
    """Create a sympy Symbol for a monomial from its name.

    Parameters
    ----------
    atomic_name : str
        The name of the monomial.

    Returns
    -------
    sympy.core.symbol.Symbol
        The corresponding symbol.
    """
    if atomic_name == "1":
        return sympy.S.One
    elif atomic_name == "0":
        return sympy.S.Zero
    else:
        return sympy.Symbol(atomic_name, commutative=True)


def symbol_prod(atomic_symbols: List[sympy.core.symbol.Symbol]
                ) -> sympy.core.mul.Mul:
    """Computes the product of a list of sympy symbols.

    Parameters
    ----------
    atomic_symbols : List[sympy.core.symbol.Symbol]
        The list of symbols to be multiplied.

    Returns
    -------
    sympy.core.mul.Mul
        The product of the symbols inputted.
    """
    prod = sympy.S.One
    for sym in atomic_symbols:
        prod *= sym
    return prod
