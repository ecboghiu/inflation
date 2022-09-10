"""
The module creates the inflation scenario associated to a causal structure. See
arXiv:1609.00672 and arXiv:1707.06476 for the original description of inflation.
@authors: Alejandro Pozas-Kerstjens, Emanuel-Cristian Boghiu
"""
import numpy as np
from itertools import chain
from warnings import warn
import methodtools
# from typing import Tuple, List, Union, NewType
from causalinflation.quantum.types import Tuple, Union, NewType

ArrayMonomial = NewType("ArrayMonomial", Union[Tuple[Tuple[int]], np.ndarray])


# from copy import deepcopy
# from types import MappingProxyType


class InflationProblem(object):
    """Class for enconding relevant details concerning the causal compatibility
    scenario.

    Parameters
    ----------
    dag : list, optional
        Dictionary where each key is a hyperedge connecting different
        parties. By default it is a single source connecting all parties.
    outcomes_per_party : list, optional
        Measurement outcome cardinalities. By default ``2`` for all parties.
    settings_per_party : list, optional
        Measurement setting cardinalities. By default ``1`` for all parties.
    inflation_level_per_source : list, optional
        Number of copies per source in the inflated graph. By default ``1`` for
        all sources.
    names : List[int], optional
        Name of each party, default is alphabetical labels, e.g.,
        ``['A', 'B', ...]``.
    verbose : int, optional
        How much information to print. By default ``0``.
    """

    def __init__(self,
                 dag=None,
                 outcomes_per_party=tuple(),
                 settings_per_party=tuple(),
                 inflation_level_per_source=tuple(),
                 order=tuple(),
                 verbose=0):
        """Initialize the InflationProblem class.
        """
        self.verbose = verbose

        if not outcomes_per_party:
            raise ValueError("Please provide outcomes per party.")
        self.outcomes_per_party = outcomes_per_party
        self.nr_parties = len(self.outcomes_per_party)
        if not settings_per_party:
            if self.verbose > 0:
                print("No settings per party provided, assuming all parties have one setting.")
            self.private_settings_per_party = [1] * self.nr_parties
        else:
            self.private_settings_per_party = settings_per_party
            assert len(
                self.private_settings_per_party) == self.nr_parties, "Different numbers of cardinalities specified for inputs versus outputs."

        # We need to infer the names of the parties. If they are explicitly given, great. Otherwise, use DAG order.
        names_have_been_set_yet = False
        if dag and (not names_have_been_set_yet):
            implicit_names_as_set = set(chain.from_iterable(dag.values()))
            assert (
                        len(implicit_names_as_set) == self.nr_parties), "DAG has a different number of outcome-associated variables than" \
                                                                        + "were given by the user-specified cardinalities."
            if order:
                sanity_check = (implicit_names_as_set.issubset(order) and implicit_names_as_set.issuperset(order))
                if sanity_check:
                    self.names = order
                    names_have_been_set_yet = True
                else:
                    warn(
                        "Names read from DAG do not match names given as keyword argument. IGNORING user-specified names.")
            if not names_have_been_set_yet:
                if len(implicit_names_as_set) > 1:
                    if self.verbose > 0:
                        warn("Order of variables is inferred by the DAG according to lexicographic order.")
                self.names = sorted(implicit_names_as_set)
                names_have_been_set_yet = True
        if order and (not names_have_been_set_yet):
            sanity_check = (len(order) == self.nr_parties)
            if sanity_check:
                self.names = order
                names_have_been_set_yet = True
            else:
                warn("Number of names does not match the user-specified cardinalities. IGNORING user-specified names.")
        if not names_have_been_set_yet:
            self.names = [chr(ord('A') + i) for i in range(self.nr_parties)]

        if not dag:
            warn("Hypergraph must be a non-empty dict of lists. Defaulting to global source.")
            self.dag = {'h_global': self.names}
        else:
            self.dag = dag

        nodes_with_children = list(dag.keys())
        # NEW PROPERTY ADDED BY ELIE
        self.split_node_model = not set(nodes_with_children).isdisjoint(self.names)
        names_to_integers_dict = {party: position for position, party in enumerate(self.names)}
        adjacency_matrix = np.zeros((self.nr_parties, self.nr_parties), dtype=np.uint8)
        for parent in nodes_with_children:
            if parent in self.names:
                ii = names_to_integers_dict[parent]
                for child in dag[parent]:
                    jj = names_to_integers_dict[child]
                    adjacency_matrix[ii, jj] = 1
        # NEW PROPERTY ADDED BY ELIE
        self.parents_per_party = list(map(np.flatnonzero, adjacency_matrix.T))
        settings_per_party_as_lists = [[s] for s in self.private_settings_per_party]
        for (party_index, party_parents_indices) in enumerate(self.parents_per_party):
            settings_per_party_as_lists[party_index].extend(np.take(self.outcomes_per_party, party_parents_indices))
        self.settings_per_party = [np.prod(multisetting) for multisetting in settings_per_party_as_lists]

        extract_parent_values_from_effective_setting = []
        for i in range(self.nr_parties):
            extract_parent_values_from_effective_setting.append(dict(zip(
                range(self.settings_per_party[i]),
                np.ndindex(tuple(settings_per_party_as_lists[i])))))
        self.extract_parent_values_from_effective_setting = extract_parent_values_from_effective_setting

        actual_sources = [source for source in nodes_with_children if source not in self.names]
        self.nr_sources = len(actual_sources)
        hypergraph = np.zeros((self.nr_sources, self.nr_parties), dtype=np.uint8)
        for ii, source in enumerate(actual_sources):
            pos = [names_to_integers_dict[party] for party in dag[source]]
            hypergraph[ii, pos] = 1

        self.hypergraph = hypergraph

        assert self.hypergraph.shape[1] == self.nr_parties, \
            (f"The number of parties derived from the DAG is {self.hypergraph.shape[1]} and " +
             f"from the specification of outcomes it is {self.nr_parties} instead")

        if np.array(inflation_level_per_source).size == 0:
            if self.verbose > 0:
                print("Inflation level per source must be a non-empty list. Defaulting to 1 (standard NPA).")
            self.inflation_level_per_source = np.array([1] * self.nr_sources)
        elif type(inflation_level_per_source) == int:
            self.inflation_level_per_source = np.array([inflation_level_per_source] * self.nr_sources)
        else:
            self.inflation_level_per_source = np.array(inflation_level_per_source)
            assert self.nr_sources == len(self.inflation_level_per_source), ("The number of sources,"
                                                                             + " as described by the hypergraph and the list of inflation levels, does not coincide")

    def __repr__(self):
        return ("InflationProblem with " + str(self.hypergraph.tolist()) +
                " as hypergraph, " + str(self.outcomes_per_party) +
                " outcomes per party, " + str(self.settings_per_party) +
                " settings per party and " +
                str(self.inflation_level_per_source) +
                " inflation copies per source.")

    @methodtools.lru_cache(maxsize=None, typed=False)
    def is_knowable_q_split_node_check(self, monomial_as_2d_numpy_array: ArrayMonomial) -> bool:
        """
        We assume that the numpy vector-per-operator notation has:
        party_index in slot 0
        outcome_index in slot -1
        effective_setting_index in slot -2
        """
        # Parties start at #1 in our numpy vector notation, so we drop by one.
        parties_in_play = np.asarray(monomial_as_2d_numpy_array)[:, 0] - 1
        # assert len(parties_in_play) == len(
        #     set(parties_in_play)), 'The same party appears to be referenced more than once.'
        parents_referenced = set()
        for p in parties_in_play:
            parents_referenced.update(self.parents_per_party[p])
        if not parents_referenced.issubset(parties_in_play):
            # Case of not an ancestrally closed set.
            return False
        # Parties start at #1 in our numpy vector notation.
        outcomes_by_party = {(o[0] - 1): o[-1] for o in monomial_as_2d_numpy_array}
        for o in monomial_as_2d_numpy_array:
            party_index = o[0] - 1
            effective_setting_as_integer = o[-2]
            o_nonprivate_settings = self.extract_parent_values_from_effective_setting[
                                        party_index][effective_setting_as_integer][1:]
            for i, p_o in enumerate(self.parents_per_party[party_index]):
                if not o_nonprivate_settings[i] == outcomes_by_party[p_o]:
                    return False
        else:
            return True

    def rectify_fake_setting_atomic_factor(self, monomial_as_2d_numpy_array: ArrayMonomial) -> ArrayMonomial:
        # Parties start at #1 in initial numpy vector notation, we reset that.
        new_mon = np.array(monomial_as_2d_numpy_array, copy=False)  # Danger.
        for o in new_mon:
            party_index = o[0] - 1
            effective_setting_as_integer = o[-2]
            o_private_settings = self.extract_parent_values_from_effective_setting[
                party_index][effective_setting_as_integer][0]
            o[-2] = o_private_settings
            # o[0] = party_index #DO NOT LOWER PARTY INDEX AS PART OF SETTING RECTIFICATION
        return new_mon


if __name__ == "__main__":
    prob = InflationProblem(dag={'U_AB': ['A', 'B'],
                                 'U_AC': ['A', 'C'],
                                 'U_AD': ['A', 'D'],
                                 'C': ['D'],
                                 'A': ['B', 'C', 'D']},
                            outcomes_per_party=(2, 2, 2, 2),
                            settings_per_party=(3, 3, 3, 3),
                            inflation_level_per_source=(1, 1, 1),
                            order=('A', 'B', 'C', 'D'),
                            verbose=2)
    print(len(prob.extract_parent_values_from_effective_setting))
    print(prob.extract_parent_values_from_effective_setting[3])
