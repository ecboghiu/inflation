"""
The module creates the inflation scenario associated to a causal structure. See
arXiv:1609.00672 and arXiv:1707.06476 for the original description of
inflation.
@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np
import sympy

from itertools import (chain,
                       combinations,
                       combinations_with_replacement,
                       product,
                       permutations)
from warnings import warn
from .sdp.fast_npa import (nb_classify_disconnected_components,
                           nb_overlap_matrix,
                           apply_source_perm,
                           commutation_matrix)

from .utils import format_permutations, partsextractor, perm_combiner
from typing import Tuple, List, Union, Dict

from functools import reduce, cached_property
from tqdm import tqdm

import networkx as nx

# Force warnings.warn() to omit the source code line in the message
# https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
import warnings
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda msg, category, filename, lineno, line=None: \
    formatwarning_orig(msg, category, filename, lineno, line="")


class InflationProblem:
    """Class for encoding relevant details concerning the causal compatibility.
    """
    def __init__(self,
                 dag: Union[Dict, None]=None,
                 outcomes_per_party: Union[Tuple[int,...], List[int], np.ndarray]=tuple(),
                 settings_per_party: Union[Tuple[int,...], List[int], np.ndarray]=tuple(),
                 inflation_level_per_source: Union[Tuple[int,...], List[int], np.ndarray]=tuple(),
                 classical_sources: Union[str, Tuple[str,...], List[str]]=tuple(),
                 intermediate_latents: Union[Tuple[str,...], List[str]]=tuple(),
                 order: Union[Tuple[str,...], List[str]]=tuple(),
                 verbose=0):
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
            it imposes that all sources are classical. By default an empty tuple.
        intermediate_latents: Tuple[str], optional
            Designates non-exogynous nodes in the DAG with these names as latent,
            in addition to all exogynous nodes.
        order : Tuple[str], optional
            Name of each party. This also fixes the order in which party outcomes
            and settings are to appear in a conditional probability distribution.
            Default is alphabetical order and labels, e.g., ``['A', 'B', ...]``.
        verbose : int, optional
            Optional parameter for level of verbose:

            * 0: quiet (default),
            * 1: monitor level: track program process and show warnings,
            * 2: debug level: show properties of objects created.
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
            implicit_names = set(map(str, chain.from_iterable(dag.values()))).difference(intermediate_latents)
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
            self.dag = {"h_global": set(self.names)}
        else:
            #Sources are nodes with children but not parents
            self.dag = {str(parent): set(map(str, children)) for
                        parent, children in dag.items()}

        # Unpacking of visible nodes with children
        nodes_with_children_as_list = list(self.dag.keys())
        nodes_with_children = set(nodes_with_children_as_list)
        self._actual_sources = sorted(
            nodes_with_children.difference(self.names, intermediate_latents),
            key=nodes_with_children_as_list.index)
        self.nr_sources = len(self._actual_sources)
        parties_with_children = nodes_with_children.intersection(self.names)


        self.has_children   = np.zeros(self.nr_parties, dtype=bool)
        self.is_network     = set(nodes_with_children).isdisjoint(self.names)
        names_to_integers = {party: position
                             for position, party in enumerate(self.names)}
        adjacency_matrix = np.zeros((self.nr_parties, self.nr_parties),
                                    dtype=bool)
        for parent in parties_with_children:
            ii = names_to_integers[parent]
            self.has_children[ii] = True
            observable_children = set(self.dag[parent])
            while observable_children:
                child = observable_children.pop()
                if child in self.names:
                    jj = names_to_integers[child]
                    adjacency_matrix[ii, jj] = True
                    continue
                assert child in intermediate_latents, f"Error, {child} is not a recognized party or intermediate latent."
                observable_children.update(self.dag[child])

        # Compute number of settings for the unpacked variables
        self.parents_per_party = list(map(np.flatnonzero, adjacency_matrix.T))
        settings_per_party_lst = [[s] for s in self.private_settings_per_party]
        for party_idx, party_parents_idxs in enumerate(self.parents_per_party):
            settings_per_party_lst[party_idx].extend(
                np.take(self.outcomes_per_party, party_parents_idxs))
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
        self.hypergraph = np.zeros((self.nr_sources, self.nr_parties),
                                   dtype=np.uint8)
        self.sources_to_check_for_party_pair_commutation = np.zeros((self.nr_parties, self.nr_parties, self.nr_sources),
                                    dtype=bool)
        for source_idx, source in enumerate(self._actual_sources):
            children = self.dag[source]
            observable_children = children.intersection(self.names)
            latent_children = children.difference(self.names)
            latents_explored_already = latent_children.copy()
            assert latent_children.issubset(intermediate_latents), f"{latent_children.difference(intermediate_latents)} are not a recognized party or intermediate latent."
            for child in observable_children:
                child_idx = names_to_integers[child]
                self.hypergraph[source_idx, child_idx] = 1
                self.sources_to_check_for_party_pair_commutation[child_idx, child_idx, source_idx] = True
            for intermediate_latent in latent_children:
                starting_children = self.dag[intermediate_latent]
                observable_descendants_via_this_latent = starting_children.intersection(self.names)
                latents_still_to_explore = starting_children.difference(self.names)
                assert latents_still_to_explore.issubset(intermediate_latents), f"{latents_still_to_explore.difference(intermediate_latents)} are not a recognized party or intermediate latent."
                assert latents_still_to_explore.isdisjoint(latents_explored_already), f"Cycle detected among intermediate latents."
                latents_explored_already.update(latents_still_to_explore)
                while latents_still_to_explore:
                    next_latent = latents_still_to_explore.pop()
                    next_children = self.dag[next_latent]
                    observable_descendants_via_this_latent.update(next_children.intersection(self.names))
                    new_latents = next_children.difference(self.names)
                    assert new_latents.issubset(intermediate_latents), f"{new_latents.difference(intermediate_latents)} are not a recognized party or intermediate latent."
                    assert new_latents.isdisjoint(latents_explored_already), f"Cycle detected among intermediate latents."
                    latents_explored_already.update(new_latents)
                for desc in observable_descendants_via_this_latent:
                    desc_idx = names_to_integers[desc]
                    self.hypergraph[source_idx, desc_idx] = 1
                    for desc2 in observable_descendants_via_this_latent:
                        desc2_idx = names_to_integers[desc2]
                        self.sources_to_check_for_party_pair_commutation[desc_idx, desc2_idx, source_idx] = True


        assert np.sum(self.hypergraph, axis=0).all(), \
            ("There appears to be a party with no sources in its past. This is not allowed.")

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
        self._astuples_dtype = np.dtype(list(zip(['']*self._nr_properties, [self._np_dtype]*self._nr_properties)))
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
        
        # Useful for LP
        self._ortho_groups_per_party = []
        for p, measurements_per_party in enumerate(self.measurements):
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
            dtype=np.intc)

        # Here we set up compatible measurements
        if classical_sources == "all":
            self._classical_sources = np.ones(self.nr_sources, dtype=bool)
        else:
            self._classical_sources = np.zeros(self.nr_sources, dtype=bool)
        if not isinstance(classical_sources, (str, type(None))):
            if classical_sources:
                for ii, source in enumerate(self._actual_sources):
                    if source in classical_sources:
                        self._classical_sources[ii] = True
        self._nonclassical_sources = np.logical_not(self._classical_sources)

        if self._nonclassical_sources.any():
            self._default_notcomm = commutation_matrix(self._lexorder,
                                                       self._nonclassical_sources,
                                                       False)
        else:
            self._default_notcomm = np.zeros(
                (self._lexorder.shape[0], self._lexorder.shape[0]),
                dtype=bool)
        self._compatible_measurements = np.invert(self._default_notcomm)
        # Use self._ortho_groups to label operators that are orthogonal as
        # incompatible as their product is zero, and they can never be 
        # observed together with non-zero probability.
        offset = 0
        for ortho_group in self._ortho_groups:
            l = len(ortho_group)
            block = np.arange(l)
            block+= offset
            self._compatible_measurements[np.ix_(block, block)] = False
            offset+=l


    def __repr__(self):
        if self._classical_sources.all():
            source_info = "All sources are classical."
        elif self._nonclassical_sources.all():
            source_info = "All sources are quantum."
        else:
            classical_sources = self._actual_sources[self._classical_sources]
            quantum_sources   = self._actual_sources[self._nonclassical_sources]
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
    def _party_positions_within_lexorder(self):
        # TODO @Elie write docstring
        offset = 0
        party_positions_within_lexorder = []
        for ortho_groups in self._ortho_groups_per_party:
            this_party_positions = []
            for ortho_group in ortho_groups:
                l = len(ortho_group)
                block = np.arange(offset, offset+l)
                this_party_positions.append(block)
                offset+=l
            party_positions_within_lexorder.append(this_party_positions)
        return party_positions_within_lexorder
    
    def _subsets_of_compatible_mmnts_per_party(self,
                                               party: int,
                                               with_last_outcome: bool = False
                                               ) -> list:
        """Find all subsets of compatible operators for a given party. They
        are returned as lists of sets of maximum length monomials (that is,
        if AB is compatible and ABC is compatible, only ABC is returned as
        AB is a subset of ABC).

        Parameters
        ----------
        party : int
            The party specifies as its position in the list of parties,
            as specified by 'order'.
        with_last_outcome : bool, optional
            Whether the compatible measurements should include those that
            have the last outcome, by default False

        Returns
        -------
        list
            List of lists of compatible operators of maximum length.
        """
        if with_last_outcome:
            _s_ = list(chain.from_iterable(self._party_positions_within_lexorder[party]))
        else:
            _s_ = []
            for positions in self._party_positions_within_lexorder[party]:
                _s_.extend(positions[:-1])
        party_compat = self._compatible_measurements[np.ix_(_s_, _s_)]
        G = nx.from_numpy_array(party_compat)
        raw_cliques = nx.find_cliques(G)
        return [partsextractor(_s_, clique) for clique in raw_cliques]

    def _generate_compatible_monomials_given_party(self,
                                                party: int,
                                                up_to_length: int = None,
                                                with_last_outcome: bool = False
                                                  ) -> np.ndarray:
        """Helper function to generate all compatible monomials given a party.
        
        While _subsets_of_compatible_mmnts_per_party returns maximum length
        compatible operators, i.e., ABC but not AB, this function returns 
        all lengths up to the specified maximum, i.e., if up_to_length=2,
        for the compatible monomial ABC, the monomials A, B, C, AB, AC, BC, ABC
        are returned.

        Parameters
        ----------
        party : int
            The party specifies as its position in the list of parties,
            as specified by 'order'.
        up_to_length : int, optional
            _description_, by default None
        with_last_outcome : bool, optional
            Whether the compatible measurements should include those that
            have the last outcome, by default False

        Returns
        -------
        2D array
            A 2D boolean array where each row is a compatible monomial,
            and the columns are the lexorder indices of the operators.
            If we have 4 operators in the lexorder, ABCD, then
            [[1, 0, 1, 0], [0, 1, 1, 0]] corresponds to AC and BC.
        """
        # The cliques will be all unique sets of compatible operators of
        # ALL lengths, given a scenario's compatibility matrix
        cliques = self._subsets_of_compatible_mmnts_per_party(
            party,
            with_last_outcome=with_last_outcome)
        max_len_clique = max([len(c) for c in cliques])
        max_length = up_to_length if up_to_length != None else max_len_clique
        if max_length > max_len_clique:
            max_length = max_len_clique
            if self.verbose > 0:
                warn("The maximum length of physical monomials required is " +
                     "larger than the maximum sequence of compatible operators")

        # Take combinations of all lengths up to the specified maximum
        # of all the cliques
        unique_subsets = {frozenset(ops) for nr_ops in range(max_length + 1)
                              for clique in cliques
                              for ops in combinations(clique, nr_ops)}
        # Sort them in ascending lexicographic order
        unique_subsets = sorted((sorted(s) for s in unique_subsets),
                                key=lambda x: (len(x), x))
        # Convert them to boolean vector encoding
        unique_subsets_as_boolvecs = np.zeros(
            (len(unique_subsets), len(self._lexorder)), dtype=bool)
        for i, mon_as_set in enumerate(unique_subsets):
            unique_subsets_as_boolvecs[i, mon_as_set] = True
        return unique_subsets_as_boolvecs

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
    def _lexorder_lookup(self) -> dict:
        """Creates helper dictionary for quick lookup of lexorder indices.

        Returns
        -------
        dict
            Mapping an operator in .tobytes() for quick lookup of its index
            in the lexorder.
        """
        return {op.tobytes(): i for i, op in enumerate(self._lexorder)}
    
    def mon_to_lexrepr(self, mon: np.ndarray) -> np.ndarray:
        ops_as_hashes = list(map(self._from_2dndarray, mon))
        try:
            return np.array(partsextractor(self._lexorder_lookup, ops_as_hashes), dtype=np.intc)
        except KeyError:
            raise Exception(f"Failed to interpret\n{mon}\n relative to specified lexorder \n{self._lexorder}.")

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
        """Map 1D array lexorder encoding of a monomial, to a string representation.

        Returns
        -------
        list
            List of the same length as lexorder, where the i-th element is the
            string representation of the i-th operator in the lexorder.
        """
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
        return np.array(as_list)

    @cached_property
    def _lexrepr_to_symbols(self):
        """For each operator in the lexorder, create a sympy symbol with the 
        same name as returned by InflationPRoblem._lexrepr_to_names()

        Returns
        -------
        list
            List of the same length as lexorder, where the i-th element is the
            string representation of the i-th operator in the lexorder.
        """
        return np.array([sympy.Symbol(name, commutative=False) for name in self._lexrepr_to_names],
                        dtype=object)

    @cached_property
    def names_to_ints(self) -> dict:
        """Map party names to integers denoting their position in the list of
        parties, as specified by 'order', offset by 1 (i.e., first party is 1).

        Used for the 2D array representation of monomials.

        Returns
        -------
        dict
            Dictionary mapping party names to integers.
        """
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
        parties_in_play = np.asarray(monomial)[:, 0] - 1
        parties_absent_bitvector    = np.ones(self.nr_parties, dtype=bool)
        parties_absent_bitvector[parties_in_play] = False
        for p in parties_in_play.flat:
            if parties_absent_bitvector[self.parents_per_party[p]].any():
                return False # Case of not an ancestrally closed set
        # Parties start at 1 in our notation
        outcomes_by_party = {(o[0] - 1): o[-1] for o in monomial}
        for o in monomial:
            party_index           = o[0] - 1
            effective_setting     = o[-2]
            o_nonprivate_settings = \
                self.effective_to_parent_settings[
                                  party_index][effective_setting][1:]
            outcomes_of_parent_parties = partsextractor(outcomes_by_party, self.parents_per_party[party_index])
            if not np.array_equal(outcomes_of_parent_parties, o_nonprivate_settings):
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
            party_index        = o[0] - 1  # Parties start at 1 our notation
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
        lexmon_factors = self.factorize_monomial_1d(self.mon_to_lexrepr(monomial_as_2darray),
                                          canonical_order=canonical_order)
        return tuple(self._lexorder[lexmon] for lexmon in lexmon_factors)

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
        if (not self.ever_factorizes) or len(monomial_as_1darray) <= 1:
            return (monomial_as_1darray,)

        inflation_indices_position = self._lexorder_for_factorization[
            monomial_as_1darray]
        unique_inflation_indices_positions, reversion_key = np.unique(inflation_indices_position, return_inverse=True)
        adj_mat = self._inflation_indices_overlap[np.ix_(unique_inflation_indices_positions, unique_inflation_indices_positions)]
        component_labels = nb_classify_disconnected_components(adj_mat)
        nof_components = component_labels.max(initial=0) + 1
        disconnected_components = tuple(monomial_as_1darray[np.take(component_labels == i, reversion_key)]
            for i in range(nof_components))

        if canonical_order:
            disconnected_components = tuple(sorted(disconnected_components,
                                                   key=lambda x: x.tobytes()))
        return disconnected_components

    ###########################################################################
    # FUNCTIONS PERTAINING TO SYMMETRY                                        #
    ###########################################################################
    def discover_lexorder_symmetries(self) -> np.ndarray:
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
            identity_perm        = np.arange(self._nr_operators, dtype=np.intc)
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
                            dtype=np.intc
                        )
                        one_source_symmetries.append(new_order)
                    except KeyError:
                        permutation_failed = True
                        pass
                lexorder_symmetries.append(np.asarray(one_source_symmetries, dtype=np.intc))
            if permutation_failed and (self.verbose > 0):
                warn("The generating set is not closed under source swaps."
                     + " Some symmetries will not be implemented.")
            return reduce(perm_combiner, lexorder_symmetries)
        else:
            return np.arange(self._nr_operators, dtype=np.intc)[np.newaxis]
        
    @cached_property
    def lexorder_symmetries(self):
        """Discover symmetries expressed as permutations of the lexorder.
        
        Returns
        -------
        numpy.ndarray[int]
            The permutations of the lexicographic order implied by the inflation
            symmetries.
        """
        return self.discover_lexorder_symmetries()
