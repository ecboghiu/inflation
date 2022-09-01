"""
The module creates the inflation scenario associated to a causal structure. See
arXiv:1609.00672 and arXiv:1707.06476 for the original description of inflation.
@authors: Alejandro Pozas-Kerstjens, Emanuel-Cristian Boghiu
"""
import numpy as np
from warnings import warn

class InflationProblem(object):
    """Class for enconding relevant details concerning the causal
    compatibibility scenario, e.g., DAG structure, number of inputs
    per party, etc.

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
        Name of each party, default is alphabetical labels, e.g. ['A', 'B', ...]
    verbose : int, optional
        How much information to print. By default ``0``.
    """
    def __init__(self, dag=[], outcomes_per_party=[], settings_per_party=[],
                 inflation_level_per_source=[], names=[], verbose=0):
        """Initialize the InflationProblem class.
        """
        self.verbose = verbose

        if not outcomes_per_party:
            raise ValueError("Please provide outcomes per party.")
        self.outcomes_per_party = outcomes_per_party
        self.nr_parties = len(self.outcomes_per_party)

        if not dag:
            warn("Hypergraph must be a non-empty list of lists. " +
                 "Defaulting to one global source.")
            self.hypergraph = np.array([[1]*self.nr_parties], dtype=np.uint8)
        else:
            self.hypergraph, self.graph_equalities = \
                                                    self._dag_to_hypergraph(dag)
            assert self.hypergraph.shape[1] == self.nr_parties, \
                ("The number of parties derived from the DAG is " +
                 f"{self.hypergraph.shape[1]} and from the specification of " +
                 f"outcomes it is {self.nr_parties} instead.")
        self.nr_sources = self.hypergraph.shape[0]

        if np.array(inflation_level_per_source).size == 0:
            if self.verbose > 0:
                print("The inflation level per source must be a non-empty list."
                      + " Defaulting to 1 (standard NPA).")
            self.inflation_level_per_source = np.array([1]*self.nr_sources)
        elif type(inflation_level_per_source) == int:
            self.inflation_level_per_source = \
                          np.array([inflation_level_per_source]*self.nr_sources)
        else:
            self.inflation_level_per_source = \
                                            np.array(inflation_level_per_source)
            assert self.nr_sources == len(self.inflation_level_per_source),    \
                  ("The number of sources, as described by the hypergraph and "
                   + "the list of inflation levels, do not coincide.")

        if not settings_per_party:
            if self.verbose > 0:
                print("No settings per party provided, assuming all parties " +
                      "have one setting.")
            self.settings_per_party = [1] * self.nr_parties
        else:
            self.settings_per_party = settings_per_party

        if not names:
            self.names = [chr(ord('A') + i) for i in range(self.nr_parties)]
        else:
            self.names = names
            assert len(self.names) == self.nr_parties, \
                ("The number of parties, as defined by the list of names and " +
                 "the hypergraph, do not coincide")

    def __repr__(self):
        return ("InflationProblem with " + str(self.hypergraph.tolist()) +
                " as hypergraph, " + str(self.outcomes_per_party) +
                " outcomes per party, " + str(self.settings_per_party) +
                " settings per party and " +
                str(self.inflation_level_per_source) +
                " inflation copies per source.")

    def _dag_to_hypergraph(self, dag):
        """The DAG is given as a dictionary of the form {parent_node:
        List[children_nodes]}. If the DAG represents a network (i.e., it is a
        bipartite graph), it computes the hypergraph representation. Otherwise,
        it applies unpacking techniques to transform the DAG into a network, and
        tracks the constraints between the nodes in the unpacked network.

        Parameters
        ----------
        dag : Dict[str, str]
            The DAG representing the causal scenario.

        Returns
        -------
        Tuple[hypergraph, constraints]
            array with the hypergraph representation of the (unpacked)
            scenario and list of relations between the nodes in the
            unpacked hypergraph

        Raises
        ------
        Exception
            Currently non-network scenarios are not implemented.
        """
        sources = dag.keys()
        is_network = all([all([source not in drain for drain in dag.values()])
                          for source in sources])
        if is_network:
            parties = sorted(set(sum(dag.values(), [])))
            party_position = {party: pos for pos, party in enumerate(parties)}
            hypergraph = np.zeros((len(sources), len(parties)), dtype=np.uint8)
            for ii, source in enumerate(sources):
                pos = [party_position[party] for party in dag[source]]
                hypergraph[ii, pos] = 1
            return hypergraph, None
        else:
            raise Exception("The handling of non-network scenarios is not " +
                            "implemented yet")
