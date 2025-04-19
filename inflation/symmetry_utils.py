"""
This file contains auxiliary functions for discovering symmetries

@authors: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens
"""

from typing import List, Union

import numpy as np

from sympy.combinatorics import Permutation, PermutationGroup
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

    original_dag_events_order = {tuple(op): i
                           for i, op in enumerate(scenario.original_dag_events)}

    ## Find the symmetries of the distribution.
    # First, build the knowable monomials of maximum length of the original
    # DAG, and their values under the distribution.
    original_dag_monomials_values = {}
    original_dag_monomials_lexboolvecs = []
    for ins in np.ndindex(*scenario.settings_per_party):
        for outs in np.ndindex(*scenario.outcomes_per_party):
            # Build the monomial corresponding to p(outs...|ins...)
            original_dag_lexboolvec = np.zeros(len(scenario.original_dag_events),
                                               dtype=bool)
            for p, (x, a) in enumerate(zip(ins, outs)):
                original_dag_lexboolvec[
                    original_dag_events_order[(p + 1, x, a)]] = True
            original_dag_monomials_lexboolvecs += [original_dag_lexboolvec]
            # Calculate its value under the distribution
            _value = distribution[(*outs, *ins)]
            _hash = original_dag_lexboolvec.tobytes()
            original_dag_monomials_values[_hash] = _value
    original_dag_monomials_lexboolvecs = np.array(
                                             original_dag_monomials_lexboolvecs)
    original_values_1d = np.array([original_dag_monomials_values[mon.tobytes()]
                                 for mon in original_dag_monomials_lexboolvecs])
    good_perms  = []
    for perm_lexorder in tqdm(scenario._all_possible_symmetries,
                              desc="Discovering distribution symmetries",
                              disable=not scenario.verbose):
        perm_orig   = lexperm_to_origperm(perm_lexorder, scenario)
        lexboolvecs = original_dag_monomials_lexboolvecs.copy()
        lexboolvecs = lexboolvecs[:, perm_orig]  # permute the columns
        new_values_1d = np.array([original_dag_monomials_values[mon.tobytes()]
                                    for mon in lexboolvecs])
        if np.allclose(new_values_1d, original_values_1d):
            good_perms += [perm_lexorder]
    if scenario.verbose > 0:
        print(f"Found {len(good_perms)} symmetries.")

    return np.array(good_perms)

def group_elements_from_generators(generators: Union[np.ndarray,
                                                List[np.ndarray]]
                                    ) -> np.ndarray:
    """
    Given a set of generators of some permutation group, return the group
    elements in lexicographic order.

    Parameters
    ----------
    generators : Union[np.ndarray, List[np.ndarray]]
        The generators of the permutation group. Each generator is a permutation
        of the indices [0...n].
    """
    G = PermutationGroup([Permutation(perm)
                          for perm in np.unique(generators, axis=0)])
    group_elements = np.array(list(G.generate_schreier_sims(af=True)))
    return group_elements[np.lexsort(np.rot90(group_elements))]

def interpret_lexorder_symmetry(perm: np.ndarray,
                                scenario: InflationProblem) -> dict:
    """Gives a human-readable form of a permutation of the lexicographic order
    of the events in the scenario.

    Parameters
    ----------
    perm : np.ndarray
        The permutation of the lexicographic order of the events in the
        scenario.
    scenario : InflationProblem
        The scenario object.

    Returns
    -------
    dict
        A dictionary that maps the names of the events in lexicographic order
        to the names of the corresponding events under the symmetry.
    """
    return dict(zip(scenario._lexrepr_to_names,
                    scenario._lexrepr_to_names[perm]))

def interpret_original_symmetry(perm: np.ndarray,
                                scenario: InflationProblem) -> dict:
    """Gives a human-readable form of a permutation of the observable events in
    the scenario.

    Parameters
    ----------
    perm : np.ndarray
        The permutation of the events in the scenario.
    scenario : InflationProblem
        The scenario object.

    Returns
    -------
    dict
        A dictionary that maps the names of the events in lexicographic order
        to the names of the corresponding events under the symmetry.
    """
    return dict(zip(scenario._original_event_names,
                    scenario._original_event_names[perm]))

def lexperm_to_origperm(lexperm: np.ndarray,
                        scenario: InflationProblem) -> np.ndarray:
    """
    Given a permutation of the lexicographic order of the events in the
    scenario, return the permutation of the observable events in the scenario.

    Parameters
    ----------
    lexperm : np.ndarray
        The permutation of the lexicographic order of the events in the
        scenario.
    scenario : InflationProblem
        The scenario object.

    Returns
    -------
    np.ndarray
        The corresponding permutation of the observable events in the scenario.
    """
    return scenario._lexidx_to_origidx[lexperm][scenario._canonical_lexids]
