import itertools
import numpy as np
import warnings
import qutip as qt
from typing import Union, List, Tuple, Dict, Literal
import sympy as sp
import cvxpy as cp
import numba as nb
from copy import deepcopy

def flatten(l: List) -> List:
    return [item for sublist in l for item in sublist]

def find_permutation(list1: List, list2: List) -> List:
    """Find the permutation that maps list1 to list2."""
    if (len(list1) != len(list2)) or (set(list1) != set(list2)):
        raise Exception('The two lists are not permutations of one another')
    else:
        original_dict = {x: i for i, x in enumerate(list1)}
        return [original_dict[x] for x in list2]
    
def generate_ops(outcomes_per_party: Dict, settings_per_party: Dict):
    """generate symbolic operators of the form partyString_input_output
    that can be used to construct bell inequalities over which we can
    optimise"""
    ops = []
    for p, nx in settings_per_party.items():
        ops_x = []
        for x in range(nx):
            ops_x.append([sp.Symbol(f'{p}_{x}_{a}')
                          for a in range(outcomes_per_party[p])])
        ops.append(ops_x)
    return ops
    
def Bell_CG2prob(belloperator, outcomes_per_party: dict, settings_per_party: dict):
    """
    Takes a Bell operator in CG as a SymPy expression built with operators
    returned by generate_ops and outputs a np.ndarray with the coefficients
    of the bell inequality encoded as output[a,b,c,...,x,y,z,...]
    """
    nr_parties = len(outcomes_per_party)
    parties = sorted(list(outcomes_per_party.keys()))
    bell_prob = np.zeros((*[outcomes_per_party[p] for p in parties],
                          *[settings_per_party[p] for p in parties]))
    sym_expanded_constant_term = 0
    template_const = np.zeros((*[outcomes_per_party[p] for p in parties],
                               *[settings_per_party[p] for p in parties]))  
    for outcomes in np.ndindex(*[outcomes_per_party[p] for p in parties]):
        sym_expanded_constant_term += np.prod([sp.symbols(f"{p}_0_{outcomes[i]}")
                                               for i, p in enumerate(parties)])
        template_const[(*outcomes,*(0,)*nr_parties)] += 1
        
    expanded_bell = 0
    belloperator = sp.expand(belloperator)
    for term, coeff in belloperator.as_coefficients_dict().items():
        # if term == 1:
        #     bell_prob += float(coeff)*template_const
        #     expanded_bell += coeff*sym_expanded_constant_term
        #     # expanded_bell += coeff
        # else:
        _, ops = term.as_coeff_mul()
        if len(ops) == nr_parties:
            outs = [int(n.split('_')[-1]) for n in str(term).split('*')]
            ins  = [int(n.split('_')[-2]) for n in str(term).split('*')]
            bell_prob[(*outs, *ins)] += float(coeff)
            expanded_bell += coeff*term
        else:
            # Add the missing operators
            present_parties = [n.split('_')[0] for n in str(term).split('*')]
            missing_parties = [p for p in parties if p not in present_parties] if present_parties else []
            expanded_term = term
            for p in missing_parties:
                expanded_term *= sum([sp.Symbol(f'{p}_0_{a}')
                                    for a in range(outcomes_per_party[p])])
            expanded_term = sp.expand(expanded_term)
            expanded_bell += coeff*expanded_term
            for term2, coeff2 in expanded_term.as_coefficients_dict().items():
                outs = [int(n.split('_')[-1]) for n in str(term2).split('*')]
                ins  = [int(n.split('_')[-2]) for n in str(term2).split('*')]
                bell_prob[(*outs, *ins)] += float(coeff)*float(coeff2)
    return bell_prob
    
    
def process_DAG(dag):
    # Build a list of Hilbert spaces, useful for getting permutations
    Hilbert_spaces_states = {q: ['H_' + p + '_' + q for p in plist]
                            for q, plist in dag.items()}
    parties = sorted(list(set([p for q in dag for p in dag[q]])))
    Hilbert_spaces_parties = {p: [] for p in parties}
    for q, plist in dag.items():
        for p in plist:
            Hilbert_spaces_parties[p].append('H_' + p + '_' + q)
    return  Hilbert_spaces_states, Hilbert_spaces_parties  


class NetworkScenario:
    def __init__(self,
                 dag: dict,
                 outcomes_per_party: dict,
                 settings_per_party: dict = None,
                 ) -> None:
        """Class for defining a causal scenario. Includes useful information
        such as outcome and setting cardinalities and permutations that map 
        the Hilbert space of the states to that of the measurements.

        Parameters
        ----------
        dag : dict
            Directed-Acyclic-Graph (DAG) describing the causal structure of the
            scenario. The keys are the states and the values are the parties
            having access to that state.
        outcomes_per_party : dict
            Dictionary with the outcomes per party. The keys are the parties
            and the values are the outcomes.
        settings_per_party : dict, optional
            Dictionary with the settings per party. The keys are the parties
            and the values are the settings. By default None (all settings 
            are 1).
        """
        self.dag = dag
        self.states  = list(dag.keys())
        self.outcomes_per_party = outcomes_per_party
        self.parties = list(outcomes_per_party.keys())
        self.nr_parties = len(self.parties)
        self.outcomes = list(outcomes_per_party.values())
        if settings_per_party is not None:
            self.settings_per_party = settings_per_party
        else:
            warnings.warn("settings_per_party not specified, setting to 1 for all parties")
            self.settings_per_party = {p: 1 for p in self.parties} 
        self.settings = list(self.settings_per_party.values())

        Hilbert_space_states, Hilbert_space_parties = process_DAG(self.dag)
        self.Hilbert_space_states  = Hilbert_space_states
        self.Hilbert_space_parties = Hilbert_space_parties
        
                
        Hilbert_space_order_states = flatten(Hilbert_space_states.values())
        Hilbert_space_order_povms  = flatten(Hilbert_space_parties.values())
        self.perm_povms2states = find_permutation(Hilbert_space_order_states,
                                                  Hilbert_space_order_povms)
        self.perm_states2povms = find_permutation(Hilbert_space_order_povms,
                                                  Hilbert_space_order_states)

        self.Hilbert_spaces = sorted(Hilbert_space_order_states + 
                                     Hilbert_space_order_povms)

def permuteHilbertSpaces(state, state_dims, perm):
    _perm = [*perm, *(len(perm) + np.array(perm)).tolist()]
    _dims = [*state_dims, *state_dims]
    state = np.transpose(np.reshape(state, _dims), axes=_perm)
    dim = int(np.sqrt(np.prod(_dims)))
    return np.reshape(state, (dim, dim))

@nb.njit(fastmath=True,parallel=True)
def nb_kron(A, B):
    out=np.empty((A.shape[0],B.shape[0],A.shape[1],B.shape[1]),dtype=A.dtype)
    for i in nb.prange(A.shape[0]):
        for j in range(B.shape[0]):
            for k in range(A.shape[1]):
                for l in range(B.shape[1]):
                    out[i,j,k,l]=A[i,k]*B[j,l]
    return out

def kronecker_product(*args):
    """Kronecker product of a list of matrices"""
    if len(args) == 1:
        return args[0]
    else:
        return np.kron(args[0], kronecker_product(*args[1:]))

def np_prob_from_states_povms(states,
                                 povms,
                                 outcomes_per_party, 
                                 settings_per_party,
                                 final_state_dims,
                                 final_povm_dims,
                                 perm_states2povms,
                                 perm_povms2states,
                                 permute_states=True):
    """Build the probability distribution form numerical states and povms.
    
        states: dictionary with source names as keys and values the states
        povms: dictionary with party names as keys and as value a matrix such that
    povms[party][setting][outcome] gives the corresponding matrix element
    for the party setting and outcome
        final_state_dims: list of dimensions of local hilbert spaces of the final state, e.g. [2,3,2,2]
        perm_states2povms: the permutation that shuffles the hilbert spaces to match the order of the POVMs
    """

    # Do the swap of the hilbert spaces to get the correct order
    # if indices are (ijk...,i'j'k'...), we have the permutation on (ijk...) 
    # but we also need to do it on (i'j'k'...). This is why we enlarge the 
    # permutation list
    # final_state_dims = [*final_state_dims, *final_state_dims]
    # final_povm_dims = [*final_povm_dims, *final_povm_dims]
    # perm_states2povms = [*perm_states2povms, *(len(perm_states2povms) + np.array(perm_states2povms)).tolist()]
    # perm_povms2states = [*perm_povms2states, *(len(perm_povms2states) + np.array(perm_povms2states)).tolist()]


    # Build the tensor product of all the states
    full_state = 1
    for state in states.values():
        full_state = np.kron(full_state, state)
        # full_state = cp.kron(full_state, state)
    
    if permute_states == True:
        full_state = permuteHilbertSpaces(full_state, final_state_dims, perm_states2povms)

    parties = list(outcomes_per_party.keys())
    prob = np.zeros((*list(outcomes_per_party.values()),
                     *list(settings_per_party.values())), dtype=object)
    for outcomes in np.ndindex(*list(outcomes_per_party.values())):
        for settings in np.ndindex(*list(settings_per_party.values())):
            measop = 1
            for i, p in enumerate(parties):
                measop = np.kron(measop, povms[p][settings[i]][outcomes[i]])
            if permute_states == False:
                measop = permuteHilbertSpaces(measop, final_povm_dims, perm_povms2states)
            prob[(*outcomes, *settings)] = np.real(np.trace(full_state @ measop))
    
    return prob

def generate_random_mixed_state(dim):
    return qt.rand_dm(dim, density=1).data.A

def generate_random_povm(dim, nr_outcomes):
    def _unitary2povm(U, nr_outcomes):
        """ Given a unitary U, return a list of nr_outcomes POVMs"""
        d = U.shape[0]
        povms = []
        V = U[:,:int(d/nr_outcomes)]  # Isometry X->X(x)Y from Unitary X(x)Y -> X(x)Y
        Vt = V.T.conj()
        for a in range(nr_outcomes):
            diag = np.zeros(nr_outcomes)
            diag[a] = 1
            id_times_proj = np.kron(np.eye(int(d/nr_outcomes)), np.diag(diag))
            povm = Vt @ id_times_proj @ V
            povms.append(povm)
        return povms
    unitary_dim = dim * nr_outcomes
    U = qt.rand_unitary(unitary_dim, density=1).data.A
    povm_effects = _unitary2povm(U, nr_outcomes)
    return povm_effects


def compute_effective_Bell_operator(objective_fullprob, 
                                    states, 
                                    povms,
                                    variable_to_optimise_over,
                                    outcomes_per_party,
                                    settings_per_party,
                                    state_support,
                                    povm_support,
                                    Hilbert_space_dims):
    outcome_cards = list(outcomes_per_party.values())
    setting_cards = list(settings_per_party.values())
    parties = list(outcomes_per_party.keys())
    party2idx = {p: i for i, p in enumerate(parties)}
    
    povms_dims = {p: np.prod([Hilbert_space_dims[h] for h in sup]) for p, sup in povms_support.items()}
    state_dims = {s: np.prod([Hilbert_space_dims[h] for h in sup]) for s, sup in state_support.items()}
    
    all_state_supports = flatten(list(state_support.values()))
    all_povm_supports = flatten(list(povm_support.values()))
    
    final_state_dims = [Hilbert_space_dims[s] for s in all_state_supports]
    
    final_povm_dims = [Hilbert_space_dims[s] for s in all_povm_supports]
    perm_povms2states = find_permutation(all_state_supports, all_povm_supports)
    perm_states2povms = find_permutation(all_povm_supports, all_state_supports)
    
    if isinstance(variable_to_optimise_over, tuple):
        # We are optimising over a POVM as the variable is e.g. ('A', 0, 2) specifies settings 0 and 2 as variables
        optimised_party, *optimised_settings = variable_to_optimise_over
        optimised_settings = set(optimised_settings)  # Make set for faster membership checking
        optimised_party_idx = party2idx[optimised_party]
        
        full_state = kronecker_product(*list(states.values()))
        
        bell_operator = np.zeros((settings_per_party[optimised_party], settings_per_party[optimised_party]),
                                    dtype=object) # I use an np.ndarray of type object so I don't need to think about the dimensions, they're a pain to keep track of
        # for a in range(2):
        #     for x in range(2):
        #         bell_operator[a,x]=[]
        for outs in np.ndindex(*outcome_cards):
            for ins in np.ndindex(*setting_cards):
                mmnts = kronecker_product(*[povms[p][ins[i]][outs[i]].astype(np.complex64)
                                            if (p != optimised_party) or (ins[optimised_party_idx] not in optimised_settings)
                                            else np.eye(povms_dims[p]).astype(np.complex64)
                                            for i, p in enumerate(parties)])
                bell_operator[outs[optimised_party_idx], ins[optimised_party_idx]] \
                    += objective_fullprob[(*outs, *ins)] * mmnts
                # bell_operator[outs[optimised_party_idx], ins[optimised_party_idx]] += [[p+str(ins[i])+str(outs[i])
                #                             if (p != optimised_party) or (ins[optimised_party_idx] not in optimised_settings)
                #                             else '1'
                #                             for i, p in enumerate(parties)]]
                # print("(a,b)=",outs,"(x,y)=",ins,[p+str(ins[i])+str(outs[i])
                #                             if (p != optimised_party) or (ins[optimised_party_idx] not in optimised_settings)
                #                             else '1'
                #                             for i, p in enumerate(parties)])
        
        # Permute the Hilbert spaces to match the order of the POVMs
        full_state = permuteHilbertSpaces(full_state, final_state_dims, perm_states2povms)
        
        for x in range(settings_per_party[optimised_party]):
            for a in range(outcomes_per_party[optimised_party]):
                bell_operator[a, x] = full_state @ bell_operator[a, x] 
                if x not in optimised_settings:
                    bell_operator[a, x] = np.real(np.trace(bell_operator[a, x]))
                else:
                    # Partial trace over the hilbert spaces that are not being optimised over
                    # TODO: use qutip for now but this could be implemented in numpy
                    bell_operator[a, x] = qt.Qobj(bell_operator[a, x], dims=(final_povm_dims, final_povm_dims))
                    bell_operator[a, x] = bell_operator[a, x].ptrace([i for i, support in enumerate(all_povm_supports)
                                                                       if support in povm_support[optimised_party]])
                    bell_operator[a, x] = bell_operator[a, x].data.A
                    # Make Hermitian to machine precision
                    bell_operator[a, x] = (bell_operator[a, x] + bell_operator[a, x].conj().T) / 2
                    ### NOT WORKING WHAT FOLLOWS
                    # _op_ = bell_operator[a, x].copy().reshape((*final_povm_dims, *final_povm_dims))
                    # axes_to_sum_over = [i for i, support in enumerate(all_povm_supports)
                    #                     if support not in povm_support[optimised_party]]
                    # axes_to_sum_over = tuple([*axes_to_sum_over, *(len(all_povm_supports) + np.array(axes_to_sum_over)).tolist()])
                    # _op_ = _op_.sum(axis=axes_to_sum_over)
                    # _op_ = _op_.reshape((povms_dims[optimised_party], povms_dims[optimised_party]))
                    # bell_operator[a, x] = _op_
    else:
        # We are optimising over a quantum state
        optimised_state = variable_to_optimise_over
        full_state = kronecker_product(*[value.astype(np.complex64) if state != optimised_state else np.eye(state_dims[state]).astype(np.complex64)
                                        for state, value in states.items()])
        bell_operator = 0
        for outs in np.ndindex(*outcome_cards):
            for ins in np.ndindex(*setting_cards):
                mmnts = kronecker_product(*[povms[p][ins[i]][outs[i]].astype(np.complex64) for i, p in enumerate(parties)])
                bell_operator += objective_fullprob[(*outs, *ins)] * mmnts
        bell_operator = permuteHilbertSpaces(bell_operator, final_povm_dims, perm_povms2states)
        bell_operator = full_state @ bell_operator
        # Partial trace with QuTip
        bell_operator = qt.Qobj(bell_operator, dims=(final_state_dims, final_state_dims))
        bell_operator = bell_operator.ptrace([i for i, support in enumerate(all_state_supports)
                                                if support in state_support[optimised_state]])
        bell_operator = bell_operator.data.A
        bell_operator = (bell_operator + bell_operator.conj().T) / 2  # Make Hermitian to machine precision
        # ### NOT WORKING WHAT FOLLOWS
        # Partial trace over the hilbert spaces that are not being optimised over
        # _op_ = bell_operator.copy().reshape((*final_state_dims, *final_state_dims))
        # axes_to_sum_over = [i for i, support in enumerate(all_state_supports) if support not in state_support[optimised_state]]
        # axes_to_sum_over = tuple([*axes_to_sum_over, *(len(all_state_supports) + np.array(axes_to_sum_over)).tolist()])
        # _op_ = _op_.sum(axis=axes_to_sum_over)
        # _op_ = _op_.reshape((state_dims[optimised_state], state_dims[optimised_state]))
        # bell_operator = _op_

    return bell_operator

def see_saw(outcomes_per_party,
            settings_per_party, 
            objective_cg,
            Hilbert_space_dims,
            fixed_states,
            fixed_povms,
            state_support,
            povms_support,
            optimisation_path=None
            ):
    """See-saw algorithm for finding the maximum value of a linear objective"""

    state_support = flatten([v.Hilbert_space_support for v in fixed_states.values()])
    assert len(set(state_support)) == len(state_support), "fixed_states must specify states with disjoint support"
    povm_support = flatten([v.Hilbert_space_support for v in fixed_povms.values()])
    assert len(set(povm_support)) == len(povm_support), "fixed_povms must specify povms with disjoint support"
    assert set(state_support) == set(povm_support), "fixed_states and fixed_povms must specify states and povms with the same support"
       
    final_state_dims = [Hilbert_space_dims[s] for s in state_support]
    final_povm_dims = [Hilbert_space_dims[s] for s in povm_support]
    perm_povms2states = find_permutation(state_support, povm_support)
    perm_states2povms = find_permutation(povm_support, state_support)
    
    povms_dims = {p: np.prod([Hilbert_space_dims[h] for h in sup]) for p, sup in povms_support.items()}
    state_dims = {s: np.prod([Hilbert_space_dims[h] for h in sup]) for s, sup in state_support.items()}
    
    outcome_cards = list(outcomes_per_party.values())
    setting_cards = list(settings_per_party.values())
    parties = list(outcomes_per_party.keys())
    party2idx = {p: i for i, p in enumerate(parties)}
       
    # Convert the objective to full probability notation, while 'objective_cg'
    # is a SymPy expression in Collins-gisin form, 'objective_fullprob' is a
    # np.ndarray with the coefficients of the Bell inequality encoded as
    # objective_fullprob[a,b,c,...,x,y,z,...]
    objective_fullprob = Bell_CG2prob(objective_cg, outcomes_per_party, settings_per_party)
    
    # Create a local copy of the fixed states and povms dictionaries
    _cvxpy_states = fixed_states.copy()
    _cvxpy_povms = fixed_povms.copy()
    
    # Create a list of CVXPY variables that we will optimise over, by checking 
    # which entries are 'None'. If an entry is 'None', add a string representing
    # it to cvxpy_variables. Then, add to _cvxpy_states and _cvxpy_povms where
    # the entry is 'None' either a random state and POVM for that setting
    cvxpy_variables = []
    for s in _cvxpy_states.keys():
        if _cvxpy_states[s] is None:
            # Keep that of this variable as we will optimise over it
            cvxpy_variables += [s]
            _cvxpy_states[s] = generate_random_mixed_state(Hilbert_space_dims[s])
        else:
            # Check that the dimension corresponds to that deduced from Hilbert_space_dims
            assert _cvxpy_states[s].shape == (state_dims[s], state_dims[s]), \
                f"Dimension of fixed state {s} does not match that deduced from Hilbert_space_dims"
    for party in parties:
        _list_of_x = []
        for x, povm in enumerate(_cvxpy_povms[party]):
            if povm is None:
                # Keep that of this variable as we will optimise over it, also
                # keep track of the setting
                _list_of_x += [x]
                _cvxpy_povms[party][x] = generate_random_povm(Hilbert_space_dims[party],
                                                                outcomes_per_party[party])
            else:
                # Check that the dimension corresponds to that deduced from Hilbert_space_dims
                assert len(_cvxpy_povms[party][x]) == outcome_cards[party], "Number of outcomes is not correct"
                assert _cvxpy_povms[party][x][0].shape == (povms_dims[party], povms_dims[party]), \
                    f"Dimension of fixed POVM {party} does not match that deduced from Hilbert_space_dims"
        cvxpy_variables += [(povm, *_list_of_x)]
       
    # Now we build precompile different parts of the optimisation problem 
    # by using CVXPY parametrisation. 
    #
    # EXAMPLE:
    #
    # b·p = \sum_abcxyz b_abcxyz tr rho_AB x rho_A'C x rho_B'C' Axa x Byb x Czc
    # 
    # If the state rho_A'C is a variable then we can write the objective as
    # 
    # b·p = tr rho_A'C (tr_ABB'C \sum_abcxyz b_abcxyz (rho_AB x Id x rho_B'C')(Axa x Byb x Czc) ) 
    #     = tr rho_A'C * PARTIAL_BELL_OPERATOR
    # 
    # where we do partial traces. If the measurement Axa is a variable then:
    #
    # b·p = \sum_ax tr Axa (tr_BB'CC' \sum_bcyz b_abcxyz (rho_AB x rho_A'C x rho_B'C')(Id x Byb x Czc) ) 
    #     = \sum_ax tr Axa * PARTIAL_BELL_OPERATOR[a, x]
    #
    # Here the shape of the SDP is a bit different, and the CVXPY parameters will need
    # to be indexed by a and x. We then need a function to compute the
    # PARTIAL_BELL_OPERATOR[a, x] for each a and x, in case of a POVM variable,
    # and just PARTIAL_BELL_OPERATOR in case of a state variable.
    
    # The function compute_effective_Bell_operator computes this PARTIAL_BELL_OPERATOR

       
    _cvxpy_states = fixed_states.copy()
    _cvxpy_povms = fixed_povms.copy()
    
    cvxpy_constraints = [] 

    prob = cp.Problem(cp.Maximize(1), cvxpy_constraints)

    prob.solve(verbose=True) # mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'})
    
    if prob.status == 'infeasible':
        print("something wrong...")
    
    # If solved correctly, go through the variables and extract the values
    # of the variable state and POVM
    
    return prob.value


def seesaw(outcomes_per_party,
           settings_per_party,
           objective_as_array,
           Hilbert_space_dims,
           fixed_states,
           fixed_povms,
           state_support,
           povms_support):
    """

    """
    parties = list(outcomes_per_party.keys())
    outcome_cards = list(outcomes_per_party.values())
    setting_cards = list(settings_per_party.keys())

    povm_dims = {p: np.prod([Hilbert_space_dims[h] for h in sup]) for p, sup
                 in povms_support.items()}
    state_dims = {s: np.prod([Hilbert_space_dims[h] for h in sup]) for s, sup
                  in state_support.items()}

    seesaw_states = fixed_states.copy()
    seesaw_povms = deepcopy(fixed_povms)

    # Create a list of CVXPY variables that we will optimise over, by checking
    # which entries are 'None'. If an entry is 'None', add a string representing
    # it to cvxpy_variables. Then, add to _cvxpy_states and _cvxpy_povms where
    # the entry is 'None' either a random state and POVM for that setting
    cvxpy_variables = []
    for s in seesaw_states.keys():
        if seesaw_states[s] is None:
            # Keep that of this variable as we will optimise over it
            cvxpy_variables += [s]
        else:
            # Check that the dimension corresponds to that deduced from Hilbert_space_dims
            assert seesaw_states[s].shape == (state_dims[s], state_dims[s]), \
                f"Dimension of fixed state {s} does not match that deduced from Hilbert_space_dims"
    for party in parties:
        _list_of_x = []
        for x, povm in enumerate(seesaw_povms[party]):
            if povm is None:
                # Keep that of this variable as we will optimise over it, also
                # keep track of the setting
                _list_of_x += [x]
            else:
                # Check that the dimension corresponds to that deduced from Hilbert_space_dims
                assert len(seesaw_povms[party][x]) == outcome_cards[
                    party], "Number of outcomes is not correct"
                assert seesaw_povms[party][x][0].shape == (
                    povm_dims[party], povm_dims[party]), \
                    f"Dimension of fixed POVM {party} does not match that deduced from Hilbert_space_dims"
        cvxpy_variables += [(party, *_list_of_x)]


    # Set up problems and their parameters for each unknown value
    cvxpy_probs = {}
    cvxpy_parameters = {}
    for variable in cvxpy_variables:
        if isinstance(variable, str):
            # Set up problem for unknown states
            state_dim = (state_dims[variable],) * 2
            BellOp = cp.Parameter(shape=state_dim, hermitian=True)
            rho = cp.Variable(shape=state_dim, hermitian=True) \
                if seesaw_states[variable] is None else seesaw_states[variable]
            # cp.Variable has nonneg argument but also having hermitian=True gives
            # ValueError: Cannot set more than one special attribute in Variable.
            constraints = [rho >> 0, cp.real(cp.trace(rho)) == 1]
            objective = cp.real(cp.trace(BellOp @ rho))
            cvxpy_probs[variable] = cp.Problem(cp.Maximize(objective),
                                               constraints)
            cvxpy_parameters[variable] = BellOp
        else:
            # Set up problem for unknown povms
            party = variable[0]
            op_dims = (settings_per_party[party], outcomes_per_party[party])
            povm_dim = povm_dims[party]
            BellOps = np.zeros(op_dims, dtype=object)
            vars_in_prob = np.zeros(op_dims, dtype=object)
            for i, j in np.ndindex(*op_dims):
                BellOps[i, j] = cp.Parameter(shape=(povm_dim,) * 2,
                                             hermitian=True)
                vars_in_prob[i, j] = cp.Variable(shape=(povm_dim,) * 2,
                                                 hermitian=True) \
                    if seesaw_povms[party][j] is None \
                    else seesaw_povms[party][j]
            constraints = []
            for row in vars_in_prob:
                constraints.append(row.sum() == np.eye(povm_dim))
                for v in row:
                    constraints.append(v >> 0)
            objective = cp.real(cp.trace(sum([BellOps[a, x] @ vars_in_prob[x][a]
                                              for x, a in
                                              np.ndindex(*op_dims)])))
            cvxpy_probs[variable] = cp.Problem(cp.Maximize(objective),
                                               constraints)
            cvxpy_parameters[variable] = BellOps

    # Arguments to compute reduced Bell operator
    args = {"objective_fullprob": objective_as_array,
            "states": seesaw_states,
            "povms": seesaw_povms,
            "outcomes_per_party": outcomes_per_party,
            "settings_per_party": settings_per_party,
            "state_support": state_support,
            "povm_support": povms_support,
            "Hilbert_space_dims": Hilbert_space_dims}

    # Generate random sample states and povms and return the best value
    best_value = 0
    for i in range(1, 6):
        print(f"ITERATION {i}")
        old_value = 0
        for s in fixed_states:
            if fixed_states[s] is None:
                args["states"][s] = generate_random_mixed_state(state_dims[s])
        for p in fixed_povms:
            for j, x in enumerate(fixed_povms[p]):
                if x is None:
                    args["povms"][p][j] = generate_random_povm(povm_dims[p],
                                                               outcomes_per_party[p])

        # Iteratively optimize until objective value converges
        for v in itertools.cycle(cvxpy_variables):
            BellOps = cvxpy_parameters[v]
            prob = cvxpy_probs[v]
            if isinstance(v, str):
                # Optimizing state
                BellOps.value = compute_effective_Bell_operator(
                    **args, variable_to_optimise_over=v)
                prob.solve(verbose=False)
                args["states"][v] = prob.variables()[0].value
            else:
                # Optimizing povms
                BellOps_values = compute_effective_Bell_operator(
                    **args, variable_to_optimise_over=v)
                for p, x in np.ndindex(settings_per_party[v[0]],
                                       outcomes_per_party[v[0]]):
                    BellOps[p, x].value = BellOps_values[p, x]
                prob.solve(verbose=False)
                vars_in_prob = prob.variables()
                for j in range(outcomes_per_party[v[0]]):
                    args["povms"][v[0]][j] = [j.value for j in
                                              vars_in_prob[:settings_per_party[v[0]]]]
                    vars_in_prob = vars_in_prob[settings_per_party[v[0]]:]
            new_value = prob.value
            print(prob.value)

            if abs(new_value - old_value) < 1e-7:
                print("CONVERGENCE")
                print(args["states"])
                print(args["povms"])
                best_value = new_value if new_value > best_value \
                    else best_value
                break
            else:
                old_value = new_value
    print("BEST VALUE: ", best_value)
    
    
if __name__ == '__main__':

    ############################# CHSH #########################################

    # dag = {'psiAB': ['A', 'B']}
    # outcomes_per_party = {'A': 2, 'B': 2}
    # settings_per_party = {'A': 2, 'B': 2}
    # scenario = NetworkScenario(dag, outcomes_per_party, settings_per_party)
    # ops = generate_ops(outcomes_per_party, settings_per_party)
    # A = [1 - 2*ops[0][x][0] for x in range(settings_per_party['A'])]
    # B = [1 - 2*ops[1][x][0] for x in range(settings_per_party['B'])]
    # CHSH = A[0]*B[0] + A[0]*B[1] + A[1]*B[0] - A[1]*B[1]
    # CHSH_array = Bell_CG2prob(CHSH, outcomes_per_party, settings_per_party)

    # # Fix local dimensions
    # LOCAL_DIM = 2
    # Hilbert_space_dims = {H: LOCAL_DIM for H in scenario.Hilbert_spaces}

    # state_support = {'psiAB': ['H_A_psiAB', 'H_B_psiAB']}
    # povms_support = {'A': ['H_A_psiAB'], 'B': ['H_B_psiAB']}

    # seesaw(outcomes_per_party=outcomes_per_party,
    #        settings_per_party=settings_per_party,
    #        objective_as_array=CHSH_array,
    #        Hilbert_space_dims=Hilbert_space_dims,
    #        fixed_states={'psiAB': None},
    #        fixed_povms={'A': [None, None], 'B': [None, None]},
    #        state_support=state_support,
    #        povms_support=povms_support)

    # bell_state = np.expand_dims(np.array([1, 0, 0, 1]), axis=1)/np.sqrt(2)
    # bell_state = bell_state @ bell_state.T.conj()
    # A0 = [state.proj().data.A for state in qt.sigmaz().eigenstates()[1]]
    # A1 = [state.proj().data.A for state in qt.sigmax().eigenstates()[1]]
    # B0 = [state.proj().data.A for state in (qt.sigmaz()+qt.sigmax()).eigenstates()[1]]
    # B1 = [state.proj().data.A for state in (qt.sigmaz()-qt.sigmax()).eigenstates()[1]]
    # _A = [A0, A1]
    # _B = [B0, B1]
    #
    # fixed_states = {'psiAB': bell_state}
    # fixed_measurements = {'A': [A0, A1],
    #                       'B': [B0, B1]}
    #
    # final_state_dims = [Hilbert_space_dims[s] for s in flatten(list(state_support.values()))]
    # final_povm_dims = [Hilbert_space_dims[s] for s in flatten(list(povms_support.values()))]
    # perm_povms2states = find_permutation(flatten(list(state_support.values())), flatten(list(povms_support.values())))
    # perm_states2povms = find_permutation(flatten(list(povms_support.values())), flatten(list(state_support.values())))
    # p = np_prob_from_states_povms(fixed_states, fixed_measurements, outcomes_per_party, settings_per_party,
    #                               final_state_dims, final_povm_dims, perm_states2povms, perm_povms2states, permute_states=False)
    #
    #
    # assert abs(p.flatten().T @ CHSH_array.flatten() - 2*np.sqrt(2))<1e-7, "2sqrt(2) is not achieved, initial mmnts are not good"
    #
    # ########## Compute reduced bell operator assuming state is the SDP variable
    #
    # BellOp_yb = compute_effective_Bell_operator(Bell_CG2prob(CHSH, outcomes_per_party, settings_per_party),
    #                                 {'psiAB': bell_state}, {'A': [A0, A1], 'B': [B0, B1]},
    #                                 'psiAB',
    #                                 outcomes_per_party,
    #                                 settings_per_party,
    #                                 state_support,
    #                                 povms_support,
    #                                 Hilbert_space_dims)
    # BellOp_yb_correct = np.zeros((4,4), dtype=np.complex64)
    # for a, b, x, y in np.ndindex(2, 2, 2, 2):
    #     BellOp_yb_correct += CHSH_array[a, b, x, y] * np.kron(_A[x][a], _B[y][b])
    # assert np.allclose(BellOp_yb_correct, BellOp_yb, atol=1e-7, rtol=1e-7), "Bell operators are not equal"
    #
    # # Solve SDP with this reduced Bell operator
    #
    # rho = cp.Variable((4,4), hermitian=True)
    # constraints = [rho >> 0, cp.real(cp.trace(rho)) == 1]
    # objective =  cp.real(cp.trace(BellOp_yb @ rho))
    # prob = cp.Problem(cp.Maximize(objective), constraints)
    # prob.solve(verbose=False)
    # assert abs(prob.value - 2*np.sqrt(2)) < 1e-7, "Optimal value is not 2sqrt(2) for when using state as variable"
    #
    # ########## Compute reduced bell operator assuming Bob's mmnts are the SDP variable
    # # tr rho Axa \otimes Byb = tr (Id_A \otimes Byb ) @ rho @ (Axa \otimes Id_B)
    # #                        = tr Byb @ tr_A (rho @ (Axa \otimes Id_B))
    # #                        = tr Byb @ BellOp_yb
    # BellOp_yb = compute_effective_Bell_operator(Bell_CG2prob(CHSH, outcomes_per_party, settings_per_party),
    #                                 {'psiAB': bell_state},
    #                                 {'A': [A0, A1],
    #                                  'B': [B0, B1]},
    #                                 ('B', 0, 1),
    #                                 outcomes_per_party,
    #                                 settings_per_party,
    #                                 state_support,
    #                                 povms_support,
    #                                 Hilbert_space_dims)
    #
    # # Computing the expression manually:
    # # tr rho Axa \otimes Byb = tr (Id_A \otimes Byb ) @ rho @ (Axa \otimes Id_B)
    # #                        = tr Byb @ tr_A (rho @ (Axa \otimes Id_B))
    # #                        = tr Byb @ BellOp_yb
    # BellOp_yb_correct = np.zeros((2,2), dtype=object)
    # for y in range(2):
    #     for b in range(2):
    #         BellOp_yb_correct[y, b] = np.zeros((4, 4), dtype=np.complex64)
    # for y in range(2):
    #     for b in range(2):
    #         for x in range(2):
    #             for a in range(2):
    #                 BellOp_yb_correct[b, y] += CHSH_array[a, b, x, y] * np.kron(_A[x][a], np.eye(2))
    #         BellOp_yb_correct[b, y] = bell_state @ BellOp_yb_correct[b, y]
    #         # Do partial trace with qutip to avoid errors, but I wouldn't use qutip for our implementation
    #         BellOp_yb_correct[b, y] = qt.Qobj(BellOp_yb_correct[b, y], dims=[[2, 2], [2, 2]])
    #         BellOp_yb_correct[b, y] = BellOp_yb_correct[b, y].ptrace(1)
    #         BellOp_yb_correct[b, y] = BellOp_yb_correct[b, y].data.A
    #         assert np.allclose(BellOp_yb[b, y], BellOp_yb_correct[b, y], atol=1e-7, rtol=1e-7), "Bell operators are not equal"
    #
    # povm_dims = {p: np.prod([Hilbert_space_dims[h] for h in sup]) for p, sup in povms_support.items()}
    # B00 = cp.Variable((povm_dims['B'],)*2, hermitian=True)
    # B01 = cp.Variable((povm_dims['B'],)*2, hermitian=True)
    # B10 = cp.Variable((povm_dims['B'],)*2, hermitian=True)
    # B11 = cp.Variable((povm_dims['B'],)*2, hermitian=True)
    # _B_ = [[B00, B01], [B10, B11]]
    # constraints = [B00 + B01 == np.eye(povm_dims['B']),
    #                B10 + B11 == np.eye(povm_dims['B']),
    #                B00 >> 0, B01 >> 0, B10 >> 0, B11 >> 0]
    # objective =  cp.real(cp.trace(sum([BellOp_yb[b, y] @ _B_[y][b] for y, b in np.ndindex(2, 2)])))
    # prob = cp.Problem(cp.Maximize(objective), constraints)
    # prob.solve(verbose=False)
    # assert abs(prob.value - 2*np.sqrt(2)) < 1e-7, "Optimal value is not 2sqrt(2) for Byb"

    # ############################# MERMIN Triangle ##############################
    
    # dag_triangle = {'psiAB': ['A', 'B'],
    #                 'psiAC': ['A', 'C'],
    #                 'psiBC': ['B', 'C']}
    # outcomes_per_party = {'A': 2, 'B': 2, 'C': 2}
    # settings_per_party = {'A': 2, 'B': 2, 'C': 2}
    
    # scenario = NetworkScenario(dag_triangle, outcomes_per_party, settings_per_party)
    
    # ops = generate_ops(outcomes_per_party, settings_per_party)
    # A = [1 - 2*ops[0][x][0] for x in range(settings_per_party['A'])]
    # B = [1 - 2*ops[1][x][0] for x in range(settings_per_party['B'])]
    # C = [1 - 2*ops[2][x][0] for x in range(settings_per_party['C'])]
    
    # # # max should be 2*sqrt(2) for dag_triangle and 4 for dag_global
    # MERMIN = A[1]*B[0]*C[0] + A[0]*B[1]*C[0] + A[0]*B[0]*C[1] - A[1]*B[1]*C[1]
    # MERMIN_as_array = Bell_CG2prob(MERMIN, outcomes_per_party, settings_per_party)
    
    # # Fix local dimensions
    # LOCAL_DIM = 2
    # Hilbert_space_dims = {H: LOCAL_DIM for H in scenario.Hilbert_spaces}
    # print(Hilbert_space_dims)
    
    # # TODO: something wrong with Hilbert_space_dims, missing objective_as_array
    
    # state_support = {'psiAB': ['H_A_psiAB', 'H_B_psiAB'],
    #                  'psiAC': ['H_A_psiAC', 'H_C_psiAC'],
    #                  'psiBC': ['H_B_psiBC', 'H_C_psiBC']}
    # povms_support = {'A': ['H_A_psiAB', 'H_A_psiAC'],
    #                  'B': ['H_B_psiBC', 'H_B_psiAB'],
    #                  'C': ['H_C_psiAC', 'H_C_psiBC']}
    
    # fixed_states = {'psiAB': None, 'psiAC': None, 'psiBC': None}
    # fixed_measurements = {'A': [None, None], 'B': [None, None], 'C': [None, None]}
    
    # seesaw(outcomes_per_party=outcomes_per_party,
    #        settings_per_party=settings_per_party,
    #        objective_as_array=MERMIN_as_array,
    #        Hilbert_space_dims=Hilbert_space_dims,
    #        fixed_states=fixed_states,
    #        fixed_povms=fixed_measurements,
    #        state_support=state_support,
    #        povms_support=povms_support)

    # print("Should be 2sqrt(2)=",
    #       see_saw(scenario, MERMIN, Hilbert_space_dims, fixed_states, fixed_measurements, state_support, povms_support))

    # ############################# MERMIN Global ##############################

    dag_global = {'psiABC': ['A', 'B', 'C']}
    outcomes_per_party = {'A': 2, 'B': 2, 'C': 2}
    settings_per_party = {'A': 2, 'B': 2, 'C': 2}

    scenario = NetworkScenario(dag_global, outcomes_per_party, settings_per_party)

    ops = generate_ops(outcomes_per_party, settings_per_party)
    A = [1 - 2*ops[0][x][0] for x in range(settings_per_party['A'])]
    B = [1 - 2*ops[1][x][0] for x in range(settings_per_party['B'])]
    C = [1 - 2*ops[2][x][0] for x in range(settings_per_party['C'])]

    # # max should be 2*sqrt(2) for dag_triangle and 4 for dag_global
    MERMIN = A[1]*B[0]*C[0] + A[0]*B[1]*C[0] + A[0]*B[0]*C[1] - A[1]*B[1]*C[1]
    MERMIN_as_array = Bell_CG2prob(MERMIN, outcomes_per_party, settings_per_party)

    # Fix local dimensions
    LOCAL_DIM = 2
    Hilbert_space_dims = {H: LOCAL_DIM for H in scenario.Hilbert_spaces}

    state_support = {'psiABC': ['H_A_psiABC', 'H_B_psiABC', 'H_C_psiABC']}
    povms_support = {'A': ['H_A_psiABC'],
                     'B': ['H_B_psiABC'],
                     'C': ['H_C_psiABC']}


    fixed_states = {'psiABC': None}
    fixed_measurements = {'A': [None, None], 'B': [None, None], 'C': [None, None]}

    seesaw(outcomes_per_party=outcomes_per_party,
           settings_per_party=settings_per_party,
           objective_as_array=MERMIN_as_array,
           Hilbert_space_dims=Hilbert_space_dims,
           fixed_states=fixed_states,
           fixed_povms=fixed_measurements,
           state_support=state_support,
           povms_support=povms_support)