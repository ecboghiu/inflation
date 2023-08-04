"""
The module creates the inflation scenario associated to a causal structure. See
arXiv:1609.00672 and arXiv:1707.06476 for the original description of
inflation.
@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np

from itertools import (chain,
                       combinations_with_replacement,
                       product,
                       permutations)
from warnings import warn
from .sdp.fast_npa import (nb_classify_disconnected_components,
                           nb_overlap_matrix,
                           apply_source_perm)

from .utils import format_permutations, partsextractor
from typing import Tuple, List, Union, Dict

from functools import reduce, cached_property
from tqdm import tqdm

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
    outcomes_per_party : [np.ndarray, List[int], Tuple[int,...]]
        Measurement outcome cardinalities.
    settings_per_party : [np.ndarray, List[int], Tuple[int,...]], optional
        Measurement setting cardinalities. By default ``1`` for all parties.
    inflation_level_per_source : [int, List[int]], optional
        Number of copies per source in the inflated graph. Source order is the
        same as insertion order in `dag`. If an integer is provided, it is used
        as the inflation level for all sources. By default ``1`` for all
        sources.
    classical_sources : Union[List[str], str], optional
        Names of the sources that are assumed to be classical. If ``'all'``,
        it imposes that all sources are classical. By default empty. 
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
                 dag: Union[Dict, None]=None,
                 outcomes_per_party: Union[Tuple[int,...], List[int], np.ndarray]=tuple(),
                 settings_per_party: Union[Tuple[int,...], List[int], np.ndarray]=tuple(),
                 inflation_level_per_source: Union[Tuple[int,...], List[int], np.ndarray]=tuple(),
                 classical_sources: Union[str, Tuple[str,...], List[str]]=tuple(),
                 order: Union[Tuple[str,...], List[str]]=tuple(),
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

        # Record expected shape of input distributions
        self.expected_distro_shape = tuple(np.hstack(
            (self.outcomes_per_party,
             self.private_settings_per_party)).tolist())

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
        self.has_children   = np.zeros(self.nr_parties, dtype=bool)
        self.is_network     = set(nodes_with_children).isdisjoint(self.names)
        names_to_integers = {party: position
                             for position, party in enumerate(self.names)}
        adjacency_matrix = np.zeros((self.nr_parties, self.nr_parties),
                                    dtype=bool)
        for parent in nodes_with_children:
            if parent in self.names:
                ii = names_to_integers[parent]
                self.has_children[ii] = True
                for child in dag[parent]:
                    jj = names_to_integers[child]
                    adjacency_matrix[ii, jj] = True
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
        self._actual_sources  = np.array(
            [source for source in nodes_with_children
                    if source not in self.names])
        self.nr_sources = len(self._actual_sources)
        self.hypergraph = np.zeros((self.nr_sources, self.nr_parties),
                                   dtype=np.uint8)
        self._nonclassical_sources, self._classical_sources = [], []
        for ii, source in enumerate(self._actual_sources):
            pos = [names_to_integers[party] for party in self.dag[source]]
            self.hypergraph[ii, pos] = 1
            if classical_sources:
                if not isinstance(classical_sources, str):
                    if source in classical_sources:
                        self._classical_sources += [ii]
                    else:
                        self._nonclassical_sources += [ii]
                else:
                    if classical_sources == "all":
                        self._classical_sources = range(self.nr_sources)
            else:
                self._nonclassical_sources += [ii]
        self._nonclassical_sources   = 1 + np.array(self._nonclassical_sources)
        self._classical_sources = 1 + np.array(self._classical_sources)

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

        # Establish internal dtype
        self._np_dtype = np.find_common_type([
            np.min_scalar_type(np.max(self.settings_per_party)),
            np.min_scalar_type(np.max(self.outcomes_per_party)),
            np.min_scalar_type(self.nr_parties + 1),
            np.min_scalar_type(np.max(self.inflation_level_per_source) + 1)],
            [])

        # Create all the different possibilities for inflation indices
        self.inflation_indices_per_party = list()
        for party in range(self.nr_parties):
            inflation_indices = list()
            active_sources =self.hypergraph[:, party]
            num_copies = np.multiply(active_sources,
                                     self.inflation_level_per_source)
            # Put non-participating and non-inflated on equal footing
            num_copies = np.maximum(num_copies, 1)
            for increase_from_base in np.ndindex(*num_copies):
                inflation_indxs = active_sources + np.array(
                    increase_from_base, dtype=self._np_dtype)
                inflation_indices.append(inflation_indxs)
            self.inflation_indices_per_party.append(
                np.vstack(inflation_indices))

        all_unique_inflation_indices = np.unique(
            np.vstack(self.inflation_indices_per_party),
            axis=0).astype(self._np_dtype)
        
        # Create hashes and overlap matrix for quick reference
        self._inflation_indices_hash = {op.tobytes(): i for i, op
                                        in enumerate(
                all_unique_inflation_indices)}
        self._inflation_indices_overlap = nb_overlap_matrix(
            np.asarray(all_unique_inflation_indices, dtype=self._np_dtype))

        # Create the measurements (formerly generate_parties)
        self._nr_properties = 1 + self.nr_sources + 2
        self.measurements = list()
        for p in range(self.nr_parties):
            O_vals = np.arange(self.outcomes_per_party[p],
                               dtype=self._np_dtype)
            S_vals = np.arange(self.settings_per_party[p],
                               dtype=self._np_dtype)
            I_vals = self.inflation_indices_per_party[p]
            # TODO: Use broadcasting instead of nested for loops
            measurements_per_party = np.empty(
                (len(I_vals), len(S_vals), len(O_vals), self._nr_properties),
                dtype=self._np_dtype)
            measurements_per_party[:, :, :, 0] = p + 1
            for i, inf_idxs in enumerate(I_vals):
                measurements_per_party[i, :, :, 1:(self.nr_sources + 1)] = \
                    inf_idxs
                for s in S_vals.flat:
                    measurements_per_party[i, s, :, -2] = s
                    for o in O_vals.flat:
                        measurements_per_party[i, s, o, -1] = o
            self.measurements.append(measurements_per_party)
        self._ortho_groups_per_party = []
        
        # Useful for LP
        for p, measurements_per_party in enumerate(self.measurements):
            _ortho_groups = []
            O_card = self.outcomes_per_party[p]
            self._ortho_groups_per_party.append(
                measurements_per_party.reshape(
                    (-1, O_card, self._nr_properties)))
        self._ortho_groups = list(chain.from_iterable(self._ortho_groups_per_party))
        self._lexorder = np.vstack(self._ortho_groups).astype(self._np_dtype)
        self._nr_operators = len(self._lexorder)

        self._lexorder_for_factorization = np.array([
            self._inflation_indices_hash[op.tobytes()]
            for op in self._lexorder[:, 1:-2]],
            dtype=int)

        # Discover the inflation symmetries
        self.inf_symmetries = self.lexorder_perms_from_inflation()



    def __repr__(self):
        if len(self._classical_sources) == self.nr_sources:
            source_info = "All sources are classical."
        elif len(self._nonclassical_sources) == self.nr_sources:
            source_info = "All sources are quantum."
        else:
            classical_sources = self._actual_sources[self._classical_sources-1]
            quantum_sources   = self._actual_sources[self._nonclassical_sources - 1]
            if len(classical_sources) > 1:
                extra_1 = "s"
                extra_2 = "are"
            else:
                extra_1 = ""
                extra_2 = "is"
            source_info = f"Source{extra_1} " + ", ".join(classical_sources) \
                + f" {extra_2} classical, and "
            if len(quantum_sources) > 1:
                extra_1 = "s"
                extra_2 = "are"
            else:
                extra_1 = ""
                extra_2 = "is"
            source_info += f"source{extra_1} " + ", ".join(quantum_sources) \
                + f" {extra_2} quantum."
        return ("InflationProblem with " + str(self.dag) +
                " as DAG, " + str(self.outcomes_per_party) +
                " outcomes per party, " + str(self.settings_per_party) +
                " settings per party and " +
                str(self.inflation_level_per_source) +
                " inflation copies per source. " + source_info)

    ###########################################################################
    # HELPER UTILITY FUNCTION                                    #
    ###########################################################################

    @cached_property
    def original_dag_events(self) -> np.ndarray:
        """
        Creates the analog of a lexorder for the original DAG.
        """
        original_dag_events = []
        for p in range(self.nr_parties):
            O_vals = np.arange(self.outcomes_per_party[p],
                               dtype=self._np_dtype)
            S_vals = np.arange(self.private_settings_per_party[p],
                               dtype=self._np_dtype)
            events_per_party = np.empty(
                (len(S_vals), len(O_vals), 3),
                dtype=self._np_dtype)
            events_per_party[::, :, 0] = p + 1
            for s in S_vals.flat:
                events_per_party[s, :, -2] = s
                for o in O_vals.flat:
                    events_per_party[s, o, -1] = o
            original_dag_events.extend(
                events_per_party.reshape((-1, 3)))
        return np.vstack(original_dag_events).astype(self._np_dtype)

    @cached_property
    def _lexorder_lookup(self):
        return {op.tobytes(): i for i, op in enumerate(self._lexorder)}
    def mon_to_lexrepr(self, mon: np.ndarray) -> np.ndarray:
        ops_as_hashes = list(map(self._from_2dndarray, mon))
        try:
            return np.array(partsextractor(self._lexorder_lookup, ops_as_hashes), dtype=int)
            # return np.array([self._lexorder_lookup[op_hash] for op_hash in ops_as_hashes], dtype=int)
        except KeyError:
            raise Exception(f"Failed to interpret\n{mon}\n relative to specified lexorder.")

    def _from_2dndarray(self, array2d: np.ndarray) -> bytes:
        """Obtains the bytes representation of an array. The library uses this
        representation as hashes for the corresponding monomials.

        Parameters
        ----------
        array2d : numpy.ndarray
            Monomial encoded as a 2D array.
        """
        return np.asarray(array2d, dtype=self._np_dtype).tobytes()

    @cached_property
    def _lexrepr_to_names(self):
        # Use expectation value notation. As ndarray for rapid multiextract.
        as_list = []
        for op in self._lexorder:
            name = self.names[op[0] - 1]
            inflation_indices_as_strs = [str(i) for i in op[1:-2]]
            if self.settings_per_party[op[0]-1] == 1:
                setting_as_str = '∅'
            else:
                setting_as_str = str(op[-2])
            outcome_as_str = str(op[-1])
            char_list = [name] + inflation_indices_as_strs + [setting_as_str, outcome_as_str]
            op_as_str = "_".join(char_list)
            as_list.append(op_as_str)
        # as_list = ["_".join([self.names[op[0] - 1]]
        #                                   + [str(i) for i in op[1:]])
        #         for op in self._lexorder]
        return np.array(as_list)

    @cached_property
    def names_to_ints(self):
        return {name: i + 1 for i, name in enumerate(self.names)}

    ###########################################################################
    # FUNCTIONS PERTAINING TO KNOWABILITY                                     #
    ###########################################################################

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

    ###########################################################################
    # FUNCTIONS PERTAINING TO FACTORIZATION                                   #
    ###########################################################################

    def factorize_monomial_2d(self,
                              monomial_as_2darray: np.ndarray,
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
        monomial_as_2darray : numpy.ndarray
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
        >>> factorised = factorize_monomial(monomial_as_2darray)
        [array([[1, 0, 1, 1, 0, 0]]),
         array([[1, 0, 3, 3, 0, 0]]),
         array([[2, 1, 0, 2, 0, 0],
                [3, 1, 4, 0, 0, 0]]),
         array([[3, 3, 5, 0, 0, 0],
                [3, 4, 5, 0, 0, 0]]),
         array([[3, 6, 6, 0, 0, 0]])]
        """
        if not self.ever_factorizes:
            return (monomial_as_2darray,)
        n = len(monomial_as_2darray)
        if n <= 1:
            return (monomial_as_2darray,)

        inflation_indices_position = [self._inflation_indices_hash[
            op.tobytes()] for op in monomial_as_2darray.astype(self._np_dtype)[:, 1:-2]]
        adj_mat = self._inflation_indices_overlap[inflation_indices_position][
            :, inflation_indices_position]

        component_labels = nb_classify_disconnected_components(adj_mat)
        disconnected_components = tuple(
            monomial_as_2darray[component_labels == i]
            for i in range(component_labels.max(initial=0) + 1))

        if canonical_order:
            disconnected_components = tuple(sorted(disconnected_components,
                                                   key=lambda x: x.tobytes()))
        return disconnected_components

    def factorize_monomial_1d(self,
                              monomial_as_1darray: np.ndarray,
                              canonical_order=False) -> Tuple[np.ndarray, ...]:
        """Split a moment/expectation value into products of moments according
        to the support of the operators within the moment. The moment is
        encoded as a 1d array representing elements of the lexorder.

        The output is a tuple of ndarrays where each array represents another
        monomial s.t. their product is equal to the original monomial.

        Parameters
        ----------
        monomial_as_1darray : numpy.ndarray
            Monomial in 1d array form.
        canonical_order: bool, optional
            Whether to return the different factors in a canonical order.

        Returns
        -------
        Tuple[numpy.ndarray]
            A tuple of ndarrays, where each array represents a picklist for an atomic monomial
            factor.
        """
        if not self.ever_factorizes:
            return (monomial_as_1darray,)

        inflation_indices_position = self._lexorder_for_factorization[monomial_as_1darray]
        if len(inflation_indices_position) <= 1:
            return (monomial_as_1darray,)

        adj_mat = self._inflation_indices_overlap[inflation_indices_position][
            :, inflation_indices_position]
        component_labels = nb_classify_disconnected_components(adj_mat)
        # print(f"DEBUG: Component labels {component_labels}")
        nof_components = component_labels.max(initial=0) + 1
        disconnected_components = tuple(monomial_as_1darray[component_labels == i]
            for i in range(nof_components))

        if canonical_order:
            disconnected_components = tuple(sorted(disconnected_components,
                                                   key=lambda x: x.tobytes()))
        return disconnected_components

    ###########################################################################
    # FUNCTIONS PERTAINING TO SYMMETRY                                        #
    ###########################################################################
    def lexorder_perms_from_inflation(self) -> np.ndarray:
        """Calculates all the symmetries pertaining to the set of generating
        monomials. The new set of operators is a permutation of the old. The
        function outputs a list of all permutations.

        Returns
        -------
        numpy.ndarray[int]
            The permutations of the lexicographic order implied by the inflation
            symmetries.
        """
        sources_with_copies = [source for source, inf_level
                               in enumerate(self.inflation_level_per_source)
                               if inf_level > 1]
        if len(sources_with_copies):
            permutation_failed = False
            lexorder_symmetries = []
            identity_perm        = np.arange(self._nr_operators, dtype=int)
            for source in tqdm(sources_with_copies,
                               disable=not self.verbose,
                               desc="Calculating symmetries   ",
                               leave=True,
                               position=0):
                one_source_symmetries = [identity_perm]
                inf_level = self.inflation_level_per_source[source]
                perms = format_permutations(list(
                    permutations(range(inf_level)))[1:])
                for permutation in perms:
                    adjusted_ops = apply_source_perm(self._lexorder,
                                                     source,
                                                     permutation)
                    try:
                        new_order = np.fromiter(
                            (self._lexorder_lookup[op.tobytes()]
                             for op in adjusted_ops),
                            dtype=int
                        )
                        one_source_symmetries.append(new_order)
                    except KeyError:
                        permutation_failed = True
                        pass
                lexorder_symmetries.append(one_source_symmetries)
            if permutation_failed and (self.verbose > 0):
                warn("The generating set is not closed under source swaps."
                     + " Some symmetries will not be implemented.")
            lexorder_symmetries = np.vstack([reduce(np.take, perms)
                                             for perms in
                                             product(*lexorder_symmetries)])
            return lexorder_symmetries
        else:
            return np.arange(self._nr_operators, dtype=int)[np.newaxis]

    def _elevate_distribution_symmetries(self, dist_syms: List) -> np.ndarray:
        """Given the action of a group on the original scenario, calculates
        the action of the group on the set of generating monomials. The
        function outputs a list of all permutations.

        Parameters
        ----------
        dist_syms : List
            Each symmetry of the original distribution is encoded as a pair of
            lists, indicating events before and after the symmetry. The final
            elements of each list is the permutation of sources.
        Returns
        -------
        numpy.ndarray[int]
            The list of all permutations of the generating columns implied by
            the inflation symmetries.
        """
        inflation_event_symmetries = []
        for sym_pair in dist_syms:
            elevated_sym_pair = []
            for sym in sym_pair:
                elevated_sym = []
                for event in sym[:-1]:
                    (party, input, output) = event
                    middle = self.inflation_indices_per_party[party - 1]
                    middle = middle[np.lexsort(np.rot90(middle[:, sym[-1]]))]
                    length = len(middle)
                    elevated_events = np.hstack((
                        np.broadcast_to(party, (length, 1)),
                        middle,
                        np.broadcast_to([input, output], (length, 2))
                    ))
                    elevated_sym.extend(elevated_events)
                elevated_sym_pair.append(elevated_sym)
            inflation_event_symmetries.append(np.array(elevated_sym_pair,
                                                       dtype=self._np_dtype))
        default_order = np.arange(len(self._lexorder))
        lexorder_symmetries = [default_order]
        for pre_action, post_action in inflation_event_symmetries:
            new_lexorder = default_order.copy()
            pre = [self._lexorder_lookup[event.tobytes()]
                   for event in pre_action]
            post = [self._lexorder_lookup[event.tobytes()]
                   for event in post_action]
            new_lexorder[pre] = new_lexorder[post]
            lexorder_symmetries.append(new_lexorder)
        lexorder_symmetries = np.vstack(lexorder_symmetries)
        return lexorder_symmetries

    def _discover_graph_automorphisms(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return a list of all party relabelling symmetries (each proceeded by
        its associated source relabelling symmetry) consistent with the
        graphical symmetries of the original DAG, subject to matching
        cardinalities of inputs and outputs for all exchanged parties, and
        subject to matching inflation levels for all exchanged sources.

        Returns
        -------
        List[Tuple[numpy.ndarray, numpy.ndarray]]
            A list of all two-element tuples. The first element in each tuple
            corresponds to a permutation of the sources, the second element
            to the permutation of parties.
        """
        nr_sources = self.nr_sources
        import networkx as nx
        from networkx.algorithms import isomorphism
        g1 = nx.DiGraph()
        g1.add_nodes_from(range(self.nr_parties+nr_sources))
        for s, bool_children in enumerate(self.hypergraph):
            for child in np.flatnonzero(bool_children):
                g1.add_edge(s, child+self.nr_sources)
        for c, parents in enumerate(self.parents_per_party):
            for p in parents:
                g1.add_edge(p+self.nr_sources, c + nr_sources)
        GMgen = isomorphism.GraphMatcher(g1, g1)
        discovered_automorphisms = list()
        for mapping in GMgen.isomorphisms_iter():
            valid_automorphism = True
            for s, inf_level in enumerate(
                    self.inflation_level_per_source.flat):
                if not valid_automorphism:
                    break
                new_s = mapping[s]
                valid_automorphism = (
                        self.inflation_level_per_source[new_s] == inf_level)
            for p, (card_in, card_out) in enumerate(zip(
                    self.settings_per_party.flat,
                    self.outcomes_per_party.flat)):
                if not valid_automorphism:
                    break
                new_p = mapping[p + self.nr_sources] - nr_sources
                valid_automorphism = (
                        (self.settings_per_party[new_p] == card_in)
                    and (self.outcomes_per_party[new_p] == card_out))
            if valid_automorphism:
                discovered_automorphisms.append((
                    np.fromiter((mapping[i] for i in
                                 range(nr_sources)),
                                dtype=int),
                     np.fromiter((mapping[i + nr_sources] - nr_sources
                                  for i in
                                  range(self.nr_parties)),
                                 dtype=int),
                     ))
        return discovered_automorphisms

    #TASK: Obtain a list of all setting relabellings,
    # and all outcome-per-setting relabellings.
    def _possible_input_output_symmetries(self) -> List[np.ndarray]:
        """
        Yields all possible setting relabellings paired with all possible
        setting-dependant outcome relabellings as
        permutations of the events on the original graph. Seperated by party,
        so that iteration will involve itertools.product.
        """
        nr_original_events = len(self.original_dag_events)
        default_events_order = np.arange(nr_original_events)
        original_dag_lookup = {op.tobytes(): i
                               for i, op in enumerate(self.original_dag_events)}
        # empty_perm = np.empty((0, nr_original_events), dtype=int)
        empty_perm = default_events_order.copy().reshape((1, nr_original_events))
        possible_syms_per_party = []
        for p, (card_in, card_out) in enumerate(zip(
            self.settings_per_party.flat,
            self.outcomes_per_party.flat)):
            possible_syms = set()
            # TEMPORARY BYPASS OF COMPLICATED STUFF: skip symmetry if party `p`
            # has a nontrivial parent, or has children.
            if self.has_children[p] or any(
                    self.outcomes_per_party[parent] > 1 for
                    parent in self.parents_per_party[p]):
                possible_syms_per_party.append(empty_perm)
                continue
            ops = np.empty((card_in, card_out, 3), dtype=self._np_dtype)
            ops[:,:, 0 ] = p + 1
            for i in range(card_in):
                ops[i, :, 1] = i
            for o in range(card_out):
                ops[:, o, 2] = o
            ops_flatish = ops.reshape((card_in * card_out, 3))
            for i_perm in permutations(range(card_in)):
                ops_copy = ops.copy()
                for old_i, new_i in enumerate(i_perm):
                    ops_copy[old_i, :, 1] = new_i
                    for o_perm in permutations(range(card_out)):
                        ops_copy[old_i, :, 2] = o_perm
                        sym = []
                        for ops_pair in zip(ops_flatish,
                                ops_copy.reshape((card_in * card_out, 3))):
                            if not np.array_equal(*ops_pair):
                                sym.append(ops_pair)
                        if len(sym) >= 2:
                            discovered_sym = frozenset((
                                (op1.tobytes(),
                                 op2.tobytes()) for op1, op2 in sym))
                            possible_syms.add(discovered_sym)
            possible_syms_as_permutations = [default_events_order]
            for sym in possible_syms:
                events_order = default_events_order.copy()
                for evnt1_hash, evnt2_hash in sym:
                    events_order[original_dag_lookup[evnt1_hash]] = \
                        original_dag_lookup[evnt2_hash]
                possible_syms_as_permutations.append(events_order)
            if len(possible_syms_as_permutations):
                possible_syms_as_permutations = np.vstack(
                    possible_syms_as_permutations).astype(int).reshape(
                    (-1, nr_original_events))
            else:
                possible_syms_as_permutations = empty_perm
            possible_syms_per_party.append(possible_syms_as_permutations)

        orig_order_perms = np.vstack([reduce(np.take, perms)
                                      for perms in
                                      product(*possible_syms_per_party)])

        return np.take(self.original_dag_events, orig_order_perms, axis=0)
