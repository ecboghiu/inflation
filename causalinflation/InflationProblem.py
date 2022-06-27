import numpy as np
from itertools import chain
from warnings import warn

class InflationProblem(object):
    """Class for enconding relevant details concerning the causal
    compatibibility scenario, e.g., DAG structure, number of inputs
    per party, etc.

    Parameters
    ----------
    dag : list, optional
        Dictionary where each key is a hyperedge connecting different
        parties, by default it is a single source connecting all parties.
    outcomes_per_party : list, optional
        Measurement outcome cardinalities, by default 2 for all parties.
    settings_per_party : list, optional
        Measurement setting cardinalities, by default 1 for all parties.
    inflation_level_per_source : list, optional
        Number of copies per source in the inflated graph, by default 1
        for all sources.
    names : List[int], optional
        Name of each party, default is alphabetical labels,
        e.g. ['A', 'B', ...]
    verbose : int, optional
        How much information to print, by default 0.
    """
    def __init__(self,  dag=[], outcomes_per_party=[], settings_per_party=[],
                        inflation_level_per_source=[], names=[], verbose=0):
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


        #We need to infer the names of the parties. If they are explicitly given, great. Otherwise, use DAG order.
        if dag:
            implicit_names_as_set = set(chain.from_iterable(dag.values()))
        else:
            implicit_names_as_set = set()
        if names:
            self.names = names
            if dag:
                assert len(implicit_names_as_set.intersection(self.names)) == len(self.implicit_names_as_set), (
                    "Names read from DAG do not match names given as keyword argument.")
        elif dag:
            self.names = sorted(implicit_names_as_set)
            warn("Assuming cardinalities are given for parties in lexicographical order.")
        else:
            self.names = [chr(ord('A') + i) for i in range(self.nr_parties)]
        assert len(self.names) == self.nr_parties, ("The number" +
                                                    " of parties, as defined by the list of names" +
                                                    " and the hypergraph, does not coincide")

        if not dag:
            warn("Hypergraph must be a non-empty list of lists. Defaulting to global source.")
            self.dag = {'h_global': self.names}
        else:
            self.dag = dag

        nodes_with_children = list(dag.keys())
        #NEW PROPERTY ADDED BY ELIE
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
            self.inflation_level_per_source = np.array([1]*self.nr_sources)
        elif type(inflation_level_per_source) == int:
            self.inflation_level_per_source = np.array([inflation_level_per_source]*self.nr_sources)
        else:
            self.inflation_level_per_source = np.array(inflation_level_per_source)
            assert self.nr_sources == len(self.inflation_level_per_source), ("The number of sources,"
                                + " as described by the hypergraph and the list of inflation levels, does not coincide")




    def __repr__(self):
        return ("InflationProblem with " + str(self.hypergraph.tolist()) +
                    " as hypergraph, " + str(self.outcomes_per_party) +
                     " outcomes per party, "+ str(self.settings_per_party) +
                     " settings per party and " +
                     str(self.inflation_level_per_source) +
                     " inflation copies per source.")

    def is_knowable_q_split_node_check(self, monomial_as_2d_numpy_array) -> bool:
        """
        We assume that the numpy vector-per-operator notation has:
        party_index in slot 0
        outcome_index in slot -1
        effective_setting_index in slot -2
        """
        parties_in_play = monomial_as_2d_numpy_array[:, 0]
        # assert len(parties_in_play) == len(
        #     set(parties_in_play)), 'The same party appears to be referenced more than once.'
        parents_referenced = set()
        for p in parties_in_play:
            parents_referenced.update(self.parents_per_party[p])
        if not parents_referenced.issubset(parties_in_play):
            #Case of not an ancestorally closed set.
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
