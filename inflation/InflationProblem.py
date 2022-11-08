"""
The module creates the inflation scenario associated to a causal structure. See
arXiv:1609.00672 and arXiv:1707.06476 for the original description of
inflation.
@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np

from itertools import chain, combinations_with_replacement
from warnings import warn
from .sdp.fast_npa import (nb_classify_disconnected_components,
                           nb_overlap_matrix)
from typing import Tuple

# Force warnings.warn() to omit the source code line in the message
# https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
import warnings
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda msg, category, filename, lineno, line=None: \
    formatwarning_orig(msg, category, filename, lineno, line="")


class InflationProblem(object):
    """Class for encoding relevant details concerning the causal compatibility
    scenario.

    Parameters
    ----------
    dag : Dict[str, List[str]], optional
        Dictionary where each key is a parent node, and the corresponding value
        is a list of the corresponding children nodes. By default it is a
        single source connecting all the parties.
    outcomes_per_party : List[int]
        Measurement outcome cardinalities.
    settings_per_party : List[int], optional
        Measurement setting cardinalities. By default ``1`` for all parties.
    inflation_level_per_source : [int, List[int]], optional
        Number of copies per source in the inflated graph. Source order is the
        same as insertion order in `dag`. If an integer is provided, it is used
        as the inflation level for all sources. By default ``1`` for all
        sources.
    order : List[str], optional
        Name of each party. This also fixes the order in which party outcomes
        and settings are to appear in a conditional probability distribution.
        Default is alphabetical order and labels, e.g., ``['A', 'B', ...]``.
    verbose : int, optional
        Optional parameter for level of verbose:

        * 0: quiet (default),
        * 1: monitor level: track program process and show warnings,
        * 2: debug level: show properties of objects created.
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

        # Read settings and outcomes
        if not outcomes_per_party:
            raise ValueError("Please provide outcomes per party.")
        self.outcomes_per_party = np.array(outcomes_per_party, dtype=int)
        self.nr_parties         = len(self.outcomes_per_party)
        if not settings_per_party:
            if self.verbose > 0:
                warn("No settings per party provided, " +
                     "assuming all parties have one setting.")
            self.private_settings_per_party = np.ones(self.nr_parties,
                                                      dtype=int)
        else:
            self.private_settings_per_party = np.asarray(settings_per_party,
                                                         dtype=int)
            assert len(self.private_settings_per_party) == self.nr_parties, \
                (f"You have specified a list of {len(outcomes_per_party)} "
                 + f"outcomes and a list of {len(settings_per_party)} inputs. "
                 + "These lists must have the same length and equal to the "
                 + "number of visible variables in the scenario.")

        # Assign names to the visible variables
        names_have_been_set_yet = False
        if dag:
            implicit_names = set(map(str, chain.from_iterable(dag.values())))
            assert len(implicit_names) == self.nr_parties, \
                ("You must provide a number of outcomes for the following "
                 + f"{len(implicit_names)} variables: {implicit_names}")
            if order:
                sanity_check = (implicit_names.issubset(order)
                                and implicit_names.issuperset(order))
                if sanity_check:
                    self.names = order
                    names_have_been_set_yet = True
                else:
                    if self.verbose > 0:
                        warn("The names read from the DAG do not match those "
                             + "read from the `order` argument. The names used"
                             + " are those read from the DAG.")
            if not names_have_been_set_yet:
                if len(implicit_names) > 1:
                    if self.verbose > 0:
                        warn("The order of variables is inferred by the DAG "
                             + "according to lexicographic order.")
                self.names = sorted(implicit_names)
                names_have_been_set_yet = True
        if order and (not names_have_been_set_yet):
            sanity_check = (len(order) == self.nr_parties)
            if sanity_check:
                self.names = tuple(map(str, order))
                names_have_been_set_yet = True
            else:
                if self.verbose > 0:
                    warn("The number of names provided does not match the "
                         + "number of variables that need a name. The names "
                         + "used are determined by the list of variables "
                         + "with outcomes.")
        if not names_have_been_set_yet:
            self.names = [chr(ord("A") + i) for i in range(self.nr_parties)]

        if not dag:
            if self.verbose > 0:
                warn("The DAG must be a non-empty dictionary with parent "
                     + "variables as keys and lists of children as values. "
                     + "Defaulting to one global source.")
            self.dag = {"h_global": self.names}
        else:
            self.dag = {str(parent): tuple(map(str, children)) for
                        parent, children in dag.items()}

        # Unpacking of visible nodes with children
        nodes_with_children = list(self.dag.keys())
        self.has_children   = np.zeros(self.nr_parties, dtype=int)
        self.is_network     = set(nodes_with_children).isdisjoint(self.names)
        names_to_integers = {party: position
                             for position, party in enumerate(self.names)}
        adjacency_matrix = np.zeros((self.nr_parties, self.nr_parties),
                                    dtype=np.uint8)
        for parent in nodes_with_children:
            if parent in self.names:
                ii = names_to_integers[parent]
                self.has_children[ii] = 1
                for child in dag[parent]:
                    jj = names_to_integers[child]
                    adjacency_matrix[ii, jj] = 1
        # Compute number of settings for the unpacked variables
        self.parents_per_party = list(map(np.flatnonzero, adjacency_matrix.T))
        settings_per_party_lst = [[s] for s in self.private_settings_per_party]
        for party_idx, party_parents_idxs in enumerate(self.parents_per_party):
            settings_per_party_lst[party_idx].extend(
                np.take(self.outcomes_per_party, party_parents_idxs)
                                                     )
        self.settings_per_party = np.asarray(
            [np.prod(multisetting) for multisetting in settings_per_party_lst],
            dtype=int)

        # Build the correspondence between effective settings and the true
        # tuple of setting values of all parents.
        effective_to_parent_settings = []
        for i in range(self.nr_parties):
            effective_to_parent_settings.append(dict(zip(
                range(self.settings_per_party[i]),
                np.ndindex(*settings_per_party_lst[i]))))
        self.effective_to_parent_settings = effective_to_parent_settings

        # Create network corresponding to the unpacked scenario
        actual_sources  = [source for source in nodes_with_children
                           if source not in self.names]
        self.nr_sources = len(actual_sources)
        self.hypergraph = np.zeros((self.nr_sources, self.nr_parties),
                                   dtype=np.uint8)
        for ii, source in enumerate(actual_sources):
            pos = [names_to_integers[party] for party in self.dag[source]]
            self.hypergraph[ii, pos] = 1

        assert self.hypergraph.shape[1] == self.nr_parties, \
            ("The number of parties derived from the DAG is "
             + f"{self.hypergraph.shape[1]} and from the specification of "
             + f"outcomes it is {self.nr_parties} instead.")

        if not inflation_level_per_source:
            if self.verbose > 0:
                warn("The inflation level per source must be a non-empty list."
                     + " Defaulting to 1 (no inflation, just NPA hierarchy).")
            self.inflation_level_per_source = np.array([1] * self.nr_sources)
        elif type(inflation_level_per_source) == int:
            self.inflation_level_per_source = \
                np.array([inflation_level_per_source] * self.nr_sources)
        else:
            self.inflation_level_per_source = \
                np.array(inflation_level_per_source)
            assert self.nr_sources == len(self.inflation_level_per_source), \
                ("The number of sources as described by the unpacked " +
                 f"hypergraph, {self.nr_sources}, and by the list of " +
                 "inflation levels specified, " +
                 f"{len(self.inflation_level_per_source)}, does not coincide.")

        # Determine if the inflation problem has a factorizing pair of parties.
        shared_sources = [np.all(np.vstack(pair), axis=0) for pair in
                          combinations_with_replacement(self.hypergraph.T, 2)]
        just_one_copy = (self.inflation_level_per_source == 1)
        self.ever_factorizes = False
        for sources_are_shared in shared_sources:
            # If for some two parties, the sources that they share in common
            # can all have different inflation levels, then there exists the
            # potential for factorization.
            if ((not np.any(sources_are_shared))
                or (not np.all(just_one_copy[sources_are_shared]))):
                self.ever_factorizes = True
                break
        # Create all the different possibilities for inflation indices
        inflation_indices = list()
        for active_sources in np.unique(self.hypergraph.T, axis=0):
            num_copies = np.multiply(active_sources,
                                     self.inflation_level_per_source)
            # Put non-participating and non-inflated on equal footing
            num_copies = np.maximum(num_copies, 1)
            for increase_from_base in np.ndindex(*num_copies):
                inflation_indxs = active_sources + np.array(increase_from_base,
                                                            dtype=np.uint8)
                inflation_indices.append(inflation_indxs)
        # Create hashes and overlap matrix for quick reference
        self._inflation_indices_hash = {op.tobytes(): i for i, op
                                        in enumerate(inflation_indices)}
        self._inflation_indices_overlap = nb_overlap_matrix(
            np.asarray(inflation_indices, dtype=np.uint8))
        assert len(self._inflation_indices_hash) == len(inflation_indices), \
            "Error: duplicated inflation index pattern."

    def __repr__(self):
        return ("InflationProblem with " + str(self.dag) +
                " as DAG, " + str(self.outcomes_per_party) +
                " outcomes per party, " + str(self.settings_per_party) +
                " settings per party and " +
                str(self.inflation_level_per_source) +
                " inflation copies per source.")

    def _is_knowable_q_non_networks(self, monomial: np.ndarray) -> bool:
        """Checks if a monomial (written as a sequence of operators in 2d array
        form) corresponds to a knowable probability. The function assumes that
        the candidate monomial already passed the preliminary knowable test
        from `inflation.sdp.quantum_tools.py`.
        If the scenario is a network, this function always returns ``True``.

        Parameters
        ----------
        monomial : numpy.ndarray
            An internal representation of a monomial as a 2d numpy array. Each
            row in the array corresponds to an operator. For each row, the
            zeroth element represents the party, the last element represents
            the outcome, the second-to-last element represents the setting, and
            the remaining elements represent inflation copies.

        Returns
        -------
        bool
            Whether the monomial can be assigned a knowable probability.
        """
        # Parties start at 1 in our notation
        parties_in_play    = np.asarray(monomial)[:, 0] - 1
        parents_referenced = set()
        for p in parties_in_play:
            parents_referenced.update(self.parents_per_party[p])
        if not parents_referenced.issubset(parties_in_play):
            # Case of not an ancestrally closed set
            return False
        # Parties start at 1 in our notation
        outcomes_by_party = {(o[0] - 1): o[-1] for o in monomial}
        for o in monomial:
            party_index           = o[0] - 1
            effective_setting     = o[-2]
            o_nonprivate_settings = \
                self.effective_to_parent_settings[
                                  party_index][effective_setting][1:]
            for i, p_o in enumerate(self.parents_per_party[party_index]):
                if not o_nonprivate_settings[i] == outcomes_by_party[p_o]:
                    return False
        else:
            return True

    def factorize_monomial(self,
                           monomial: np.ndarray,
                           canonical_order=False) -> Tuple[np.ndarray, ...]:
        """Split a moment/expectation value into products of moments according
        to the support of the operators within the moment. The moment is
        encoded as a 2d array where each row is an operator. If
        ``monomial=A*B*C*B`` then row 1 is ``A``, row 2 is ``B``, row 3 is
        ``C``, and row 4 is ``B``. In each row, the columns encode the
        following information:

          * First column: The party index, *starting from 1* (e.g., 1 for
            ``A``, 2 for ``B``, etc.)
          * Last two columns: The input ``x`` starting from zero, and then the
            output ``a`` starting from zero.
          * In between: This encodes the support of the operator. There are as
            many columns as sources/quantum states. Column `j` represents
            source `j-1` (-1 because the first column represents the party). If
            the value is 0, then this operator does not measure this source. If
            the value is, e.g., 2, then this operator is acting on copy 2 of
            source `j-1`.

        The output is a tuple of ndarrays where each array represents another
        monomial s.t. their product is equal to the original monomial.

        Parameters
        ----------
        monomial : numpy.ndarray
            Monomial in 2d array form.
        canonical_order: bool, optional
            Whether to return the different factors in a canonical order.

        Returns
        -------
        Tuple[numpy.ndarray]
            A tuple of ndarrays, where each array represents an atomic monomial
            factor.
        Examples
        --------
        >>> monomial = np.array([[1, 0, 1, 1, 0, 0],
                                 [2, 1, 0, 2, 0, 0],
                                 [1, 0, 3, 3, 0, 0],
                                 [3, 3, 5, 0, 0, 0],
                                 [3, 1, 4, 0, 0, 0],
                                 [3, 6, 6, 0, 0, 0],
                                 [3, 4, 5, 0, 0, 0]])
        >>> factorised = factorize_monomial(monomial)
        [array([[1, 0, 1, 1, 0, 0]]),
         array([[1, 0, 3, 3, 0, 0]]),
         array([[2, 1, 0, 2, 0, 0],
                [3, 1, 4, 0, 0, 0]]),
         array([[3, 3, 5, 0, 0, 0],
                [3, 4, 5, 0, 0, 0]]),
         array([[3, 6, 6, 0, 0, 0]])]
        """
        if not self.ever_factorizes:
            return (monomial,)
        n = len(monomial)
        if n <= 1:
            return (monomial,)

        inflation_indices_position = [self._inflation_indices_hash[
            op.tobytes()] for op in monomial.astype(np.uint8)[:, 1:-2]]
        adj_mat = self._inflation_indices_overlap[inflation_indices_position][
            :, inflation_indices_position]

        component_labels = nb_classify_disconnected_components(adj_mat)
        disconnected_components = tuple(
            monomial[component_labels == i]
            for i in range(component_labels.max(initial=0) + 1))

        if canonical_order:
            disconnected_components = tuple(sorted(disconnected_components,
                                                   key=lambda x: x.tobytes()))
        return disconnected_components

    def rectify_fake_setting(self, monomial: np.ndarray) -> np.ndarray:
        """When constructing the monomials in a non-network scenario, we rely
        on an internal representation of operators where the integer denoting
        the setting actually is an 'effective setting' that encodes, in
        addition to the 'private setting' that each party is free to choose,
        the values of all the parents of the variable in question, which also
        effectively act as settings. This function resets this 'effective
        setting' integer to the true 'private setting' integer. It is useful to
        relate knowable monomials to their meaning as conditional events in
        non-network scenarios. If the scenario is a network, this function does
        nothing.

        Parameters
        ----------
            monomial : numpy.ndarray
                An internal representation of a monomial as a 2d numpy array.
                Each row in the array corresponds to an operator. For each row,
                the zeroth element represents the party, the last element
                represents the outcome, the second-to-last element represents
                the setting, and the remaining elements represent inflation
                copies.

        Returns
        -------
        numpy.ndarray
            The monomial with the index of the setting representing only the
            private setting.
        """
        new_mon = np.array(monomial, copy=False)
        for o in new_mon:
            party_index        = o[0] - 1     # Parties start at 1 our notation
            effective_setting  = o[-2]
            o_private_settings = \
                self.effective_to_parent_settings[
                                   party_index][effective_setting][0]
            o[-2] = o_private_settings
        return new_mon
