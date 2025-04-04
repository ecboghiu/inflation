"""
This file contains auxiliary functions for discovering symmetries

@authors: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens
"""

import numpy as np

from tqdm import tqdm
from . import InflationProblem

def discover_distribution_symmetries(distribution: np.array,
                                     scenario: InflationProblem
                                     ) -> np.ndarray:
    """
    """
    # Sanity checks
    parties = scenario.nr_parties
    if not isinstance(distribution, np.ndarray):
        raise TypeError("The distribution must be encoded in a numpy array.")
    if len(distribution.shape) != 2*parties:
        raise ValueError("The distribution must be encoded as an " + 
                         "2*nr_parties-dimensional array.")
    if np.any(distribution.shape[:parties] != scenario.outcomes_per_party):
        raise ValueError("The number of outcomes of the distribution and of " +
                         "the scenario do not match.")
    if np.any(distribution.shape[parties:] != scenario.settings_per_party):
        raise ValueError("The number of settings of the distribution and of " +
                         "the scenario do not match.")
    if not np.all(distribution >= 0):
        raise ValueError("The distribution contains negative values.")
    if not np.all(np.isclose(distribution.sum(axis=tuple(range(parties))), 1)):
        raise ValueError("The distribution is not normalized for each setting.")

    original_dag_events_order = {tuple(op): i for i, op in enumerate(scenario.original_dag_events)}

    ## Find the symmetries of the distribution.
    # First, build the knowable monomials of maximum length of the original
    # DAG, and their values under the distribution.
    original_dag_monomials_values = {}
    original_dag_monomials_lexboolvecs = []
    for ins in np.ndindex(*scenario.settings_per_party):
        for outs in np.ndindex(*scenario.outcomes_per_party):
            # Build the monomial corresponding to p(outs...|ins...)
            original_dag_lexboolvec = np.zeros(len(scenario.original_dag_events), dtype=bool)
            for p, (x, a) in enumerate(zip(ins, outs)):
                original_dag_lexboolvec[original_dag_events_order[(p + 1, x, a)]] = True
            original_dag_monomials_lexboolvecs += [original_dag_lexboolvec]
            # Calculate its value under the distribution
            _value = distribution[(*outs, *ins)]
            _hash = original_dag_lexboolvec.tobytes()
            original_dag_monomials_values[_hash] = _value
    original_dag_monomials_lexboolvecs = np.array(original_dag_monomials_lexboolvecs)
    original_values_1d = np.array([original_dag_monomials_values[mon.tobytes()]
                                for mon in original_dag_monomials_lexboolvecs])
    good_perms  = []
    for perm_lexorder in tqdm(scenario._all_possible_symmetries,
                              desc="Discovering distribution symmetries",
                              disable=not scenario.verbose):
        perm_orig = scenario.lexperm_to_origperm(perm_lexorder)
        lexboolvecs = original_dag_monomials_lexboolvecs.copy()
        lexboolvecs = lexboolvecs[:, perm_orig]  # permute the columns
        new_values_1d = np.array([original_dag_monomials_values[mon.tobytes()]
                                    for mon in lexboolvecs])
        if np.allclose(new_values_1d, original_values_1d):
            good_perms += [perm_lexorder]
    if scenario.verbose > 0:
        print(f"Found {len(good_perms)} symmetries.")

    return np.array(good_perms)
