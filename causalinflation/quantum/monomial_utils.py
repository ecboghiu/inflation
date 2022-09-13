import numpy as np
from typing import List
from collections import Counter
import sympy


def compute_marginal(prob_array: np.ndarray, atom: np.ndarray) -> float:
    """Function which, given an atomic monomial and a probability distribution prob_array
        called as prob_array[a,b,c,...,x,y,z,...], returns the numerical value of the
        probability associated to the monomial.
        The atomic monomial is a list of length-3 vectors.
        The first element indicates the party,
        the second element indicates the setting,
        the final element indicates the outcome.
        Note that this accepts marginals and then
        automatically computes all the summations over p[a,b,c,...,x,y,z,...].
        Parameters
        ----------

        prob_array : np.ndarray
            The probability distribution of dims
            (outcomes_per_party, settings_per_party).
        atom : np.ndarray
            Monomial indicated a (commuting) collection of measurement operators.
        Returns
        -------
        float
            The value of the symbolic probability (which can be a marginal)
        """
    if len(atom):
        n_parties: int = prob_array.ndim // 2
        participating_parties = atom[:, 0] - 1  # Convention in numpy monomial format is first party = 1
        inputs = atom[:, -2].astype(int)
        outputs = atom[:, -1].astype(int)
        indices_to_sum = list(set(range(n_parties)).difference(participating_parties))
        marginal_dist = np.sum(prob_array, axis=tuple(indices_to_sum))
        input_list: np.ndarray = np.zeros(n_parties, dtype=int)
        input_list[participating_parties] = inputs
        outputs_inputs = np.concatenate((outputs, input_list))
        return marginal_dist[tuple(outputs_inputs)]
    else:
        return 1.





def symbol_from_atomic_name(atomic_name: str) -> sympy.core.symbol.Symbol:
    if atomic_name == '1':
        return sympy.S.One
    elif atomic_name == '0':
        return sympy.S.Zero
    else:
        return sympy.Symbol(atomic_name, commutative=True)


def name_from_atomic_names(atomic_names: List[str]) -> str:
    # #
    if len(atomic_names):
        output = ''
        for i, (name, power) in enumerate(Counter(atomic_names).items()):
            if i > 0:
                output += '*'
            output += name
            if power > 1:
                output += '**' + str(power)
        return output
    else:
        return '1'


# def symbol_from_atomic_names(atomic_names: List[str]):
#     prod = sympy.S.One
#     for op_name in atomic_names:
#         if op_name != '1':
#             prod *= sympy.Symbol(op_name,
#                                  commutative=True)
#     return prod


def symbol_from_atomic_symbols(atomic_symbols: List[sympy.core.symbol.Symbol]) -> sympy.core.symbol.Symbol:
    prod = sympy.S.One
    for sym in atomic_symbols:
        prod *= sym
    return prod