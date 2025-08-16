"""
The module creates the inflation scenario associated to a causal structure. See
arXiv:1609.00672 and arXiv:1707.06476 for the original description of
inflation.

@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import warnings
from collections import defaultdict
from functools import reduce, cached_property
from itertools import (chain,
                       combinations_with_replacement,
                       permutations)
from typing import Tuple, List, Union, Dict
from warnings import warn

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism
from sympy import Symbol
from tqdm import tqdm
from .sdp.fast_npa import (nb_classify_disconnected_components,
                           nb_overlap_matrix,
                           apply_source_perm,
                           commutation_matrix)
from .symmetry_utils import group_elements_from_generators
from .utils import (format_permutations,
                    partsextractor,
                    perm_combiner,
                    all_and_maximal_cliques)
from .cliques_with_symmetry import all_and_maximal_cliques_symmetry

# Force warnings.warn() to omit the source code line in the message
# https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
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
                 classical_sources: Union[str, Tuple[str,...], List[str], None]=tuple(),
                 nonclassical_intermediate_latents: Union[Tuple[str,...], List[str]]=tuple(),
                 classical_intermediate_latents: Union[Tuple[str,...], List[str]]=tuple(),
                 order: Union[Tuple[str,...], List[str]]=tuple(),
                 really_just_one_source: bool=True,
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
        nonclassical_intermediate_latents: Tuple[str], optional
            Designates non-exogynous nodes in the DAG with these names as
            nonclassical latent, in addition to all exogynous nodes other than
            those specifically indicates as classical.
        classical_intermediate_latents: Tuple[str], optional
            Designates non-exogynous nodes in the DAG with these names as
            a classical-type latent, alongside any latents from classical_sources.
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

        # Internal record of intermediate latents
        self.classical_intermediate_latents = set(map(str,classical_intermediate_latents))
        self.nonclassical_intermediate_latents = set(map(str,nonclassical_intermediate_latents))
        assert self.classical_intermediate_latents.isdisjoint(self.nonclassical_intermediate_latents), "An intermediate latent cannot be both classical and nonclassical."
        self.intermediate_latents = self.classical_intermediate_latents.union(self.nonclassical_intermediate_latents)

        self.really_just_one_source = really_just_one_source

        # Assign names to the visible variables
        names_have_been_set_yet = False
        if dag:
            implicit_names = set(map(str, chain.from_iterable(dag.values()))).difference(self.intermediate_latents)
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
            # Sources are nodes with children but not parents
            self.dag = {str(parent): set(map(str, children)) for
                        parent, children in dag.items()}


        # Distinguishing between classical versus nonclassical sources
        nodes_with_children_as_list = list(self.dag.keys())
        nodes_with_children = set(nodes_with_children_as_list)
        self._actual_sources = np.asarray(sorted(
            nodes_with_children.difference(self.names, self.intermediate_latents),
            key=nodes_with_children_as_list.index), dtype=object)
        self.nr_sources = len(self._actual_sources)
        if isinstance(classical_sources, str):
            if classical_sources.lower() == "all":
                self._classical_sources = np.ones(self.nr_sources, dtype=bool)
            else:
                raise ValueError(f'The keyword argument classical_sources=`{classical_sources}` could not be parsed.')
        else:
            self._classical_sources = np.zeros(self.nr_sources, dtype=bool)
        if not isinstance(classical_sources, (str, type(None))):
            assert set(classical_sources).issubset(self._actual_sources), "Some specified classical source cannot be found in the DAG."
            for ii, source in enumerate(self._actual_sources):
                if source in classical_sources:
                    self._classical_sources[ii] = 1
        self._nonclassical_sources = np.logical_not(self._classical_sources)

        self._inverse_dag = defaultdict(set)
        for v, children in self.dag.items():
            for child in children:
                self._inverse_dag[child].add(v)
        for il in self.intermediate_latents:
            this_ils_parents = self._inverse_dag[il]
            if not this_ils_parents.isdisjoint(self.names):
                raise NotImplementedError(
                    "InflationProblem cannot handle intermediate latents with observable parents at this time.")
            elif this_ils_parents.isdisjoint(self._actual_sources.flat):
                raise ValueError(
                    f"The intermediate latent {il} has no source parent. Please add a source parent to the DAG and re-initialize InflationProblem.")

        for ncil in self.nonclassical_intermediate_latents:
            this_ils_parents = self._inverse_dag[ncil]
            if this_ils_parents.isdisjoint(self._actual_sources[self._nonclassical_sources].flat):
                raise NotImplementedError(
                    f"The nonclassical intermediate latent {ncil} has no nonclassical source parent." +
                    f"\nIts parents are {this_ils_parents} while the set of nonclassical sources is {self._actual_sources[self._nonclassical_sources]})" +
                    "\nPlease add a nonclassical source parent to the DAG and re-initialize InflationProblem.")

                # if type(inflation_level_per_source) == int:
                #     old_inflation_level = [inflation_level_per_source] * self.nr_sources
                # elif len(inflation_level_per_source) == self.nr_sources:
                #     old_inflation_level = list(inflation_level_per_source)
                # elif inflation_level_per_source:
                #     raise ValueError("inflation_level_per_source must be a list with the same length as the number of sources.")
                # new_inflation_level = old_inflation_level + [1]
                # new_dag = self.dag.copy()
                # new_dag[f'rho_single_child_{ncil}'] = [ncil]
                # return InflationProblem(dag=new_dag,
                #                         outcomes_per_party=outcomes_per_party,
                #                         settings_per_party=settings_per_party,
                #                         inflation_level_per_source=new_inflation_level,
                #                         classical_sources=classical_sources,
                #                         nonclassical_intermediate_latents=nonclassical_intermediate_latents,
                #                         classical_intermediate_latents=classical_intermediate_latents,
                #                         order=order,
                #                         verbose=verbose)


        # Unpacking of visible nodes with children
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
            if not observable_children.issubset(self.names):
                raise NotImplementedError("At this time InflationProblem does not accept DAGs with observed nodes pointing to intermediate latents.")
            latents_behind_this_node = self._inverse_dag[parent].difference(self.names)
            siblings_of_parent = set([])
            for latent in latents_behind_this_node:
                siblings_of_parent.update(self.dag[latent])
            if not observable_children.issubset(siblings_of_parent):
                raise NotImplementedError(
                    "At this time InflationProblem does not accept DAGs with directed edges between observed nodes lacking a common latent parent.")
            child_indices = [names_to_integers[child] for child in observable_children]
            adjacency_matrix[ii, child_indices] = True

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
        # We need to distinguish between quantum versus classical sources.
        # We use the internal convention that +1 indicates a shared classical source
        # whereas +2 indicates a shared quantum sources
        self.sources_to_check_for_party_pair_commutation = np.zeros((self.nr_parties, self.nr_parties, self.nr_sources),
                                    dtype=np.uint8)
        for source_idx, source in enumerate(self._actual_sources):
            quantum_source_bonus = self._nonclassical_sources[source_idx]
            children = self.dag[source]
            observable_children = children.intersection(self.names)
            latent_children = children.difference(self.names)
            assert latent_children.issubset(self.intermediate_latents), f"{latent_children.difference(self.intermediate_latents)} are not all a recognized party or an intermediate latent."
            for child in observable_children:
                child_idx = names_to_integers[child]
                self.hypergraph[source_idx, child_idx] = 1
                self.sources_to_check_for_party_pair_commutation[child_idx, child_idx, source_idx] = 1 + quantum_source_bonus
            for intermediate_latent in latent_children:
                observable_descendants_via_this_latent = self.dag[intermediate_latent]
                assert observable_descendants_via_this_latent.issubset(self.names), "At this time InflationProblem does not accept DAGs with intermediate latents pointing to other intermediate latents."
                quantum_connection_bonus = np.logical_and(intermediate_latent in self.nonclassical_intermediate_latents, quantum_source_bonus).astype(np.uint8)
                for desc in observable_descendants_via_this_latent:
                    desc_idx = names_to_integers[desc]
                    self.hypergraph[source_idx, desc_idx] = 1
                    for desc2 in observable_descendants_via_this_latent:
                        desc2_idx = names_to_integers[desc2]
                        not_accounting_for_quantum_bonus = int(self.sources_to_check_for_party_pair_commutation[desc_idx, desc2_idx, source_idx])
                        self.sources_to_check_for_party_pair_commutation[desc_idx, desc2_idx, source_idx] = max(
                            not_accounting_for_quantum_bonus,
                            1 + quantum_connection_bonus)


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
        just_one_copy = np.asarray(self.inflation_level_per_source == 1)
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
        self._np_dtype = np.result_type(*[
            np.min_scalar_type(np.max(self.settings_per_party)),
            np.min_scalar_type(np.max(self.outcomes_per_party)),
            np.min_scalar_type(self.nr_parties + 1),
            np.min_scalar_type(np.max(self.inflation_level_per_source) + 1)])

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

        self._all_unique_inflation_indices = np.unique(
            np.vstack(self.inflation_indices_per_party),
            axis=0).astype(self._np_dtype)
        
        # Create hashes and overlap matrix for quick reference
        self._inflation_indices_hash = {op.tobytes(): i for i, op
                                        in enumerate(
                self._all_unique_inflation_indices)}
        if really_just_one_source:
            overlap_matrix = self.one_source_overlap_matrix
        else:
            overlap_matrix = nb_overlap_matrix
        self._inflation_indices_overlap = overlap_matrix(
            np.asarray(self._all_unique_inflation_indices, dtype=self._np_dtype))

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

        self.measurements_symbolic = \
            [np.apply_along_axis(self._1d_to_symbol, -1, measurements_per_party)
             for measurements_per_party in self.measurements]

        # Useful for LP
        self._ortho_groups_per_party = []
        for p, measurements_per_party in enumerate(self.measurements):
            O_card = self.outcomes_per_party[p]
            self._ortho_groups_per_party.append(
                measurements_per_party.reshape(
                    (-1, O_card, self._nr_properties)))
        self._ortho_groups = list(chain.from_iterable(self._ortho_groups_per_party))

        #New - orthogonal indices. To be used to accelerate monomial instantiation.
        offset = 0
        self._ortho_idxs_per_party = []
        self._ortho_idxs = []
        for ortho_groups_of_party in self._ortho_groups_per_party:
            ortho_idxs_of_party = []
            for ortho_group in ortho_groups_of_party:
                l = len(ortho_group)
                block = np.arange(l)
                block += offset
                ortho_idxs_of_party.append(block)
                offset += l
            self._ortho_idxs_per_party.append(np.array(ortho_idxs_of_party))
            self._ortho_idxs.extend(ortho_idxs_of_party)
        self._template_idxs = np.array([ortho_idx_group[0] for ortho_idx_group in self._ortho_idxs], dtype=int)

        self._lexorder = np.vstack(self._ortho_groups).astype(self._np_dtype)
        self.party_from_lexidx = self._lexorder[:,0]
        self.party_from_templateidx = self.party_from_lexidx[self._template_idxs]
        self._nr_operators = len(self._lexorder)

        self._lexorder_for_factorization = np.array([
            self._inflation_indices_hash[op.tobytes()]
            for op in self._lexorder[:, 1:-2]],
            dtype=np.intc)

        # Set up compatible measurements
        if self._nonclassical_sources.any():
            self._default_notcomm = commutation_matrix(self._lexorder,
                                                       self.sources_to_check_for_party_pair_commutation,
                                                       False)
        else:
            self._default_notcomm = np.zeros(
                (self._lexorder.shape[0], self._lexorder.shape[0]),
                dtype=bool)

        _lexorder_to_original = self.rectify_fake_setting(
            self._lexorder[:, [0, -2, -1]])
        (self.original_dag_events,
         self._canonical_lexids,
         self._lexidx_to_origidx) = np.unique(_lexorder_to_original,
                                              return_index=True,
                                              return_inverse=True, axis=0)

        # Symmetries implied by the inflation
        if really_just_one_source:
            self.symmetries = self.inflation_symmetries_one_source
        else:
            self.symmetries = self.inflation_symmetries


    @property
    def _compatible_template_measurements(self):
        return np.invert(self._default_notcomm[np.ix_(self._template_idxs, self._template_idxs)])

    @property
    def compatible_template_symmetries(self):
        return np.unique(np.argsort(self.symmetries[:, self._template_idxs], axis=1), axis=0)

    def all_and_maximal_compatible_templates(self,
                                             max_n=0,
                                             isolate_maximal=True):
        # return all_and_maximal_cliques(self._compatible_template_measurements,
        #                                max_n=max_n,
        #                                isolate_maximal=isolate_maximal)
        return all_and_maximal_cliques_symmetry(self._compatible_template_measurements,
                                                self.compatible_template_symmetries,
                                                max_n=max_n,
                                                isolate_maximal=isolate_maximal)


    def __repr__(self):
        if self._classical_sources.all():
            source_info = "All sources are classical."
        elif self._nonclassical_sources.all():
            source_info = "All sources are quantum."
        else:
            classical_sources = self._actual_sources[self._classical_sources.astype(bool)]
            quantum_sources   = self._actual_sources[self._nonclassical_sources.astype(bool)]
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
    # HELPER UTILITY FUNCTION                                                 #
    ###########################################################################
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
    def _any_inflation(self) -> bool:
        return np.any(self.inflation_level_per_source > 1)

    @cached_property
    def _lexrepr_to_dicts(self) -> np.ndarray:
        """Map 1D array lexorder encoding of a monomial, to a string representation.

        Returns
        -------
        numpy.ndarray
            Array of the same length as lexorder, where the i-th element is the
            dictionary interpretation of the i-th operator in the lexorder.
        """
        return np.array([self._interpret_operator(op) for op in self._lexorder])

    @cached_property
    def _lexrepr_to_names(self) -> np.ndarray:
        """Map 1D array lexorder encoding of a monomial, to a string representation.

        Returns
        -------
        numpy.ndarray
            Array of the same length as lexorder, where the i-th element is the
            string representation of the i-th operator in the lexorder.
        """
        return np.asarray([self._interpretation_to_name(
            op_dict,
            include_copy_indices=self._any_inflation)
                for op_dict in self._lexrepr_to_dicts.flat], dtype=object)

    @cached_property
    def _original_event_names(self) -> np.ndarray:
        """Map 1D array lexorder-like encoding of an event, to a string representation.

        Returns
        -------
        numpy.ndarray
            Array of the same length as original events, where the i-th element is the
            string representation of the i-th operator in the event list.
        """
        return np.asarray([self._interpretation_to_name(
            self._interpret_operator(event),
            include_copy_indices=False)
                for event in self.original_dag_events], dtype=object)

    @cached_property
    def _lexrepr_to_copy_index_free_names(self) -> np.ndarray:
        """Map 1D array lexorder encoding of a monomial, to a string representation.

        Returns
        -------
        numpy.ndarray
            Array of the same length as lexorder, where the i-th element is the
            string representation of the i-th operator in the lexorder.
        """
        if not self._any_inflation:
            return self._lexrepr_to_names
        else:
            return np.asarray([self._interpretation_to_name(
                op_dict,
                include_copy_indices=False)
                for op_dict in self._lexrepr_to_dicts.flat], dtype=object)

    @cached_property
    def _lexrepr_to_all_names(self) -> np.ndarray:
        """Map 1D array lexorder encoding of a monomial to
        a variety of possible string representations.

        Returns
        -------
        numpy.ndarray
            Array of the same length as lexorder, where the i-th element is a
            collection of string representations of the i-th operator in the
            lexorder.
        """
        old_names_v1 = [self._interpretation_to_name_old(op_dict,
                                                         replace_trivial_setting=True)
                        for op_dict in self._lexrepr_to_dicts.flat]
        old_names_v2 = [legacy_name.replace('∅','0') for legacy_name in old_names_v1]
        return np.stack((
            self._lexrepr_to_names,
            self._lexrepr_to_copy_index_free_names,
            old_names_v1,
            old_names_v2
        ), 1).astype(object)

    @cached_property
    def _lexrepr_to_symbols(self) -> np.ndarray:
        """For each operator in the lexorder, create a sympy symbol with the 
        same name as returned by InflationProblem._lexrepr_to_names()

        Returns
        -------
        list
            List of the same length as lexorder, where the i-th element is the
            string representation of the i-th operator in the lexorder.
        """
        return np.array([Symbol(name, commutative=False)
                         for name in self._lexrepr_to_names],
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

    def _1d_to_symbol(self, repr_1d: np.ndarray) -> Symbol:
        """Convert a 1D representation of an operator to a sympy symbol.
        
        Parameters
        ----------
        repr_1d : np.ndarray
            1D representation of an operator.

        Returns
        -------
        sympy.Symbol
            Sympy symbol representing the operator.
        """
        name = self._interpretation_to_name(self._interpret_operator(repr_1d))
        if not self._any_inflation:
            # If there is no inflation, we can remove the copy indices
            name = name.split('^{')[0] + name.split('}')[1]
        return Symbol(name, commutative=False)

    ###########################################################################
    # FUNCTIONS PERTAINING TO KNOWABILITY                                     #
    ###########################################################################
    def _interpret_operator(self, op: np.ndarray) -> dict:
        interpretation = {}
        party_int = op[0] -1
        interpretation["Party as Integer"] = party_int
        interpretation["Party"] = self.names[party_int]
        outcome_int = op[-1]
        interpretation["Outcome"] = outcome_int
        setting_as_single_int = op[-2]
        interpretation["Composite Setting"] = setting_as_single_int
        interpretation["Composite Setting is Trivial"] = (self.settings_per_party[party_int] == 1)
        setting_as_tuple = self.effective_to_parent_settings[party_int][setting_as_single_int]
        interpretation["Setting as Tuple"] = setting_as_tuple
        private_setting = setting_as_tuple[0]
        interpretation["Private Setting"] = private_setting
        interpretation["Private Setting is Trivial"] = (self.private_settings_per_party[party_int] == 1)
        outcomes_of_parents = setting_as_tuple[1:]
        parents_in_play_as_ints = self.parents_per_party[party_int]
        interpretation["Parents in-play as Integers"] = parents_in_play_as_ints
        parents_in_play_as_names = partsextractor(self.names, parents_in_play_as_ints)
        non_private_setting_dict = dict(zip(parents_in_play_as_names, outcomes_of_parents))
        interpretation["Do Values"] = non_private_setting_dict
        if len(op)==3:
            return interpretation
        interpretation["Copy Indices"] = op[1:-2]
        relevant_slots = np.logical_and(
            self.inflation_level_per_source > 0,
            interpretation["Copy Indices"] > 0
        )
        interpretation["Relevant Copy Indices"] = interpretation["Copy Indices"][relevant_slots]
        return interpretation

    @staticmethod
    def _make_interpretation_hashable(op_as_dict):
        return (int(op_as_dict["Party as Integer"]),
                tuple(int(i) for i in op_as_dict["Copy Indices"]),
                tuple(int(i) for i in op_as_dict["Setting as Tuple"]),
                int(op_as_dict["Outcome"]))


    def _interpretation_to_name(self, op: dict, include_copy_indices=True) -> str:
        op_as_str = op["Party"]
        if not op["Private Setting is Trivial"]:
            op_as_str += '_'+str(op["Private Setting"])
        if include_copy_indices or self.really_just_one_source:
            if len(op["Relevant Copy Indices"]):
                copy_index_string = '^{'
                copy_index_string += ','.join(map(str,op["Relevant Copy Indices"].flat))
                copy_index_string += '}'
                op_as_str += copy_index_string
            # copy_indices_string = "_" + "_".join(map(str, op["Copy Indices"]))
            # op_as_str += copy_indices_string
        op_as_str += f"={op['Outcome']}"
        if len(op["Do Values"]):
            do_values_string = "|do("
            do_values_string += ','.join((f"{p}={v}" for p, v in op["Do Values"].items()))
            do_values_string += ')'
            op_as_str += do_values_string
        return op_as_str

    @staticmethod
    def _interpretation_to_name_old(op: dict, replace_trivial_setting=True) -> str:
        op_as_str = op["Party"]
        copy_indices_string = "_" + "_".join(map(str, op["Copy Indices"]))
        op_as_str += copy_indices_string
        if replace_trivial_setting and op["Composite Setting is Trivial"]:
            op_as_str += "_∅"
        else:
            op_as_str += "_" + str(op["Composite Setting"])
        op_as_str += "_" + str(op['Outcome'])
        return op_as_str

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
        >>> factorised = InflationProblem.factorize_monomial_2d(monomial_as_2darray)
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
    # FUNCTIONS PERTAINING TO SYMMETRIES                                      #
    ###########################################################################
    @cached_property
    def inflation_symmetries(self) -> np.ndarray:
        """Calculates all the symmetries pertaining to the set of generating
        monomials due to copy index relabelling. The new set of operators is a
        permutation of the old. The function outputs a list of all permutations.

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
            symmetries         = []
            identity_perm      = np.arange(self._nr_operators, dtype=np.intc)
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
                symmetries.append(np.asarray(one_source_symmetries,
                                             dtype=np.intc))
            if permutation_failed and (self.verbose > 0):
                warn("The generating set is not closed under source swaps."
                     + " Some symmetries will not be implemented.")
            return reduce(perm_combiner, symmetries)
        return np.arange(self._nr_operators, dtype=np.intc)[np.newaxis]

    def add_symmetries(self,
                       new_symmetries: Union[np.ndarray, List[np.ndarray]]
                       ):
        """Adds new symmetries, represented as permutations of the
        lexicographical order, to the list of symmetries in the scenario.

        Parameters
        -------
        new_symmetries : numpy.ndarray[int], List[np.ndarray[int]]
            The permutations of the lexicographic order that conform the
            symmetries to be included.
        """
        self.symmetries = group_elements_from_generators(
            np.vstack((self.symmetries, new_symmetries)))

    def reset_symmetries(self):
        """Remove all the symmetries of the scenario, keeping only the ones
            that are implied by the inflation structure.        
        """
        self.symmetries = self.inflation_symmetries

    @cached_property
    def _lexorder_hashable_interpretation_decoder(self):
        return {self._make_interpretation_hashable(op_as_dict): i for
                    i, op_as_dict in enumerate(self._lexrepr_to_dicts)}

    @cached_property
    def _party_relabelling_symmetries(self) -> np.ndarray:
        """Return a list of all party relabelling symmetries (each proceeded by
        its associated source relabelling symmetry) consistent with the
        graphical symmetries of the original DAG, subject to matching
        cardinalities of inputs and outputs for all exchanged parties, and
        subject to matching inflation levels for all exchanged sources.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            The first array gives permutations of the lexorder, the second array
            gives the permutations of the original events.
        """
        nr_sources = self.nr_sources
        g1 = nx.DiGraph()
        g1.add_nodes_from(range(self.nr_parties+nr_sources))
        for s, bool_children in enumerate(self.hypergraph):
            for child in np.flatnonzero(bool_children):
                g1.add_edge(s, child+self.nr_sources)
        for c, parents in enumerate(self.parents_per_party):
            for p in parents:
                g1.add_edge(p+self.nr_sources, c + nr_sources)
        GMgen = isomorphism.DiGraphMatcher(g1, g1)
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

        lexorder_perms = []
        for automorphism in discovered_automorphisms:
            source_perm, party_perm = automorphism
            template = self._lexorder.copy()
            for p, new_p in zip(range(self.nr_parties), party_perm):
                template[self._lexorder[:, 0] == p + 1, 0] = new_p + 1
            new_source_perm = np.argsort(source_perm)
            template = template[:, [0]+(1+new_source_perm).tolist() + [-2, -1]]
            lexorder_perm = np.array([self._lexorder_lookup[op.tobytes()]
                                      for op in template])
            lexorder_perms += [lexorder_perm]
        return np.array(lexorder_perms)

    @cached_property
    def _setting_specific_outcome_relabelling_symmetries(self) -> np.ndarray:
        """
        Yields all possible setting relabellings paired with all possible
        setting-dependant outcome relabellings as permutations of the events on
        the original graph. Seperated by party, so that iteration will involve
        itertools.product.
        """
        identity_perm = np.arange(self._nr_operators, dtype=int)
        sym_generators = [identity_perm]

        for p in range(self.nr_parties):
            # We do not attempt outcome relabelling on parties with children
            # and parents!
            if self.has_children[p] and (self.settings_per_party[p] > 1):
                break
            for x in range(self.private_settings_per_party[p]):
                for i, perm in enumerate(
                                   permutations(
                                       range(self.outcomes_per_party[p]))):
                    if i == 0:
                        continue  # skip empty perm
                    new_interpretations = [op_as_dict.copy()
                                           for op_as_dict
                                           in self._lexrepr_to_dicts]
                    lexorder_perm = []
                    for op_as_dict in new_interpretations:
                        if p in op_as_dict["Parents in-play as Integers"]:
                            to_adjust = list(op_as_dict["Setting as Tuple"])
                            old_value = to_adjust[p+1]
                            new_value = perm[old_value]
                            to_adjust[p+1] = new_value
                            op_as_dict["Setting as Tuple"] = tuple(to_adjust)
                        if ((op_as_dict["Party as Integer"] == p)
                            and (op_as_dict["Private Setting"] == x)):
                            old_value = op_as_dict["Outcome"]
                            op_as_dict["Outcome"] = perm[old_value]
                        lexorder_perm.append(
                            self._lexorder_hashable_interpretation_decoder[
                                self._make_interpretation_hashable(op_as_dict)])
                    lexorder_perm = np.array(lexorder_perm)
                    sym_generators.append(lexorder_perm)
        return np.array(sym_generators)

    @cached_property
    def _party_specific_setting_relabelling_symmetries(self) -> np.ndarray:
        """
        Yields all possible setting relabellings that are specific to just one
        party as permutations of the events on the original graph. Seperated by
        party, so that iteration will involve itertools.product.
        """
        identity_perm = np.arange(self._nr_operators, dtype=int)
        sym_generators = [identity_perm]
        for p in range(self.nr_parties):
            # Since we are only adjusting PRIVATE setting, we can proceed even
            # if the party has children
            for i, perm in enumerate(
                            permutations(
                                range(self.private_settings_per_party[p]))):
                if i == 0:
                    continue  # skip empty perm
                new_interpretations = [op_as_dict.copy() for op_as_dict in
                                       self._lexrepr_to_dicts]
                lexorder_perm = []
                for op_as_dict in new_interpretations:
                    if op_as_dict["Party as Integer"] == p:
                        to_adjust = list(op_as_dict["Setting as Tuple"])
                        old_value = to_adjust[0]
                        new_value = perm[old_value]
                        to_adjust[0] = new_value
                        op_as_dict["Setting as Tuple"] = tuple(to_adjust)
                    lexorder_perm.append(
                        self._lexorder_hashable_interpretation_decoder[
                            self._make_interpretation_hashable(op_as_dict)])
                lexorder_perm = np.array(lexorder_perm)
                sym_generators.append(lexorder_perm)
        return np.array(sym_generators)

    @cached_property
    def _all_possible_symmetries(self) -> np.ndarray:
        group_generators = np.vstack((
            self._party_relabelling_symmetries,
            self._party_specific_setting_relabelling_symmetries,
            self._setting_specific_outcome_relabelling_symmetries))
        group_elements = group_elements_from_generators(group_generators)
        return group_elements

    ###########################################################################
    # FUNCTIONS PERTAINING TO ACTUALLY ONE SOURCE                             #
    ###########################################################################
    @staticmethod
    def exists_shared_source_modified(inf_indices1: np.ndarray,
                                inf_indices2: np.ndarray) -> bool:
        common_sources = np.logical_and(inf_indices1, inf_indices2)
        if not np.any(common_sources):
            return False
        return not set(inf_indices1[common_sources]).isdisjoint(set(inf_indices2[common_sources]))


    def one_source_overlap_matrix(self, all_inflation_indxs: np.ndarray) -> np.ndarray:
        n = len(all_inflation_indxs)
        adj_mat = np.eye(n, dtype=bool)
        for i in range(1, n):
            inf_indices_i = all_inflation_indxs[i]
            for j in range(i):
                inf_indices_j = all_inflation_indxs[j]
                if self.exists_shared_source_modified(inf_indices_i, inf_indices_j):
                    adj_mat[i, j] = True
        adj_mat = np.logical_or(adj_mat, adj_mat.T)
        return adj_mat

    @cached_property
    def inflation_symmetries_one_source(self) -> np.ndarray:
        """Calculates all the symmetries pertaining to the set of generating
        monomials due to copy index relabelling. The new set of operators is a
        permutation of the old. The function outputs a list of all permutations.

        Returns
        -------
        numpy.ndarray[int]
            The permutations of the lexicographic order implied by the inflation
            symmetries.
        """
        assert np.array_equiv(self.inflation_level_per_source,
                              self.inflation_level_per_source[0]), """
                              Only call this with uniform inflation level!"""
        inf_level = max(self.inflation_level_per_source)
        identity_perm = np.arange(self._nr_operators, dtype=np.intc)
        symmetries = [identity_perm]
        if inf_level>1:
            permutation_failed = False
            perms = format_permutations(list(
                permutations(range(inf_level)))[1:])
            all_sources_simultanous = np.arange(len(self.inflation_level_per_source))
            for permutation in perms:
                adjusted_ops = apply_source_perm(self._lexorder,
                                                 all_sources_simultanous,
                                                 permutation)
                try:
                    new_order = np.fromiter(
                        (self._lexorder_lookup[op.tobytes()]
                         for op in adjusted_ops),
                        dtype=np.intc
                    )
                    symmetries.append(new_order)
                except KeyError:
                    permutation_failed = True
            if permutation_failed and (self.verbose > 0):
                warn("The generating set is not closed under source swaps."
                     + " Some symmetries will not be implemented.")
        return group_elements_from_generators(symmetries)
