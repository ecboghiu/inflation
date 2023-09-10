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

def seesaw(outcomes_per_party,
           settings_per_party,
           objective_as_array,
           Hilbert_space_dims,
           fixed_states,
           fixed_povms,
           state_support,
           povms_support,
           seesaw_max_iterations=10,
           verbose=1):
    parties = list(outcomes_per_party.keys())
    party2idx = {p: i for i, p in enumerate(parties)}
    outcome_cards = list(outcomes_per_party.values())
    setting_cards = list(settings_per_party.values())

    povm_dims = {p: np.prod([Hilbert_space_dims[h] for h in sup]) for p, sup
                 in povms_support.items()}
    state_dims = {s: np.prod([Hilbert_space_dims[h] for h in sup]) for s, sup
                  in state_support.items()}
    all_state_supports = flatten(list(state_support.values()))
    all_povm_supports = flatten(list(povms_support.values()))    
    final_state_dims = [Hilbert_space_dims[s] for s in all_state_supports]
    final_povm_dims = [Hilbert_space_dims[s] for s in all_povm_supports]
    perm_povms2states = find_permutation(all_state_supports, all_povm_supports)
    perm_states2povms = find_permutation(all_povm_supports, all_state_supports)

    seesaw_states = deepcopy(fixed_states)
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
                    party2idx[party]], "Number of outcomes is not correct"
                assert seesaw_povms[party][x][0].shape == (
                    povm_dims[party], povm_dims[party]), \
                    f"Dimension of fixed POVM {party} does not match that deduced from Hilbert_space_dims"
        # cvxpy_variables += [(party, *_list_of_x)]
        for x in _list_of_x:
            cvxpy_variables += [(party, x)]


    # Set up problems and their parameters for each unknown value
    cvxpy_probs = {}
    cvxpy_parameters = {}
    cvxpy_name2variable = {}
    for variable in cvxpy_variables:
        if isinstance(variable, str):
            # Set up problem for unknown states
            state_dim = (state_dims[variable],) * 2
            BellOp = cp.Parameter(shape=state_dim, hermitian=True)
            rho = cp.Variable(shape=state_dim, hermitian=True, name=str(variable)) \
                if seesaw_states[variable] is None else seesaw_states[variable]
            cvxpy_name2variable[rho.name()] = rho
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
            for x, a in np.ndindex(*op_dims):
                BellOps[a, x] = cp.Parameter(shape=(povm_dim,) * 2,
                                             hermitian=True)
                if fixed_povms[party][x] is None:
                    vars_in_prob[x, a] = cp.Variable(shape=(povm_dim,) * 2,
                                                    hermitian=True,
                                                    name=str(variable)+'_'+str(a)+str(x))
                    cvxpy_name2variable[vars_in_prob[x, a].name()] = \
                        vars_in_prob[x, a]
                else:
                    vars_in_prob[x, a] = seesaw_povms[party][x][a]
                    
            constraints = []
            for row in vars_in_prob:
                summ = row.sum()
                if not isinstance(summ, np.ndarray):
                    constraints.append(summ == np.eye(povm_dim))
                for v in row:
                    if not isinstance(v, np.ndarray):
                        constraints.append(v >> 0)
            objective = cp.real(cp.trace(sum([BellOps[a, x] @ vars_in_prob[x, a]
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
    for i in range(1, seesaw_max_iterations):
        print(f"ITERATION {i}. Choosing random initial states and POVMs.")
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
        LOOP_LIMIT = 100
        for loop_index in range(LOOP_LIMIT):
            # for v in itertools.cycle(cvxpy_variables):
            for v in cvxpy_variables:
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
                        **args, variable_to_optimise_over=(v[0],*range(settings_per_party[v[0]])))
                    for x, a in np.ndindex(settings_per_party[v[0]],
                                        outcomes_per_party[v[0]]):
                        BellOps[a, x].value = BellOps_values[a, x]
                    prob.solve(verbose=False)
                    # vars_in_prob = prob.variables()
                    for x in v[1:]:
                        args["povms"][v[0]][x] = [cvxpy_name2variable[str(v)+'_'+str(a)+str(x)].value for a in
                                                range(outcomes_per_party[v[0]])]
                        # args["povms"][v[0]][x] = [Pa.value for Pa in
                        #                           vars_in_prob[:settings_per_party[v[0]]]]
                        # vars_in_prob = vars_in_prob[settings_per_party[v[0]]:]
                new_value = prob.value
                if verbose > 1:
                    if new_value < old_value:
                        print(f"WARNING: VALUE DECREASED by {abs(old_value-new_value)}")
                
            if verbose:
                print(f"Sweep {loop_index}/{LOOP_LIMIT}: {prob.value}")

            if abs(new_value - old_value) < 1e-7:
                if verbose:
                    print("CONVERGENCE")
                if new_value > best_value:
                    best_value = new_value
                    best_states = deepcopy(args["states"])
                    # clean states
                    for k, v in best_states.items():
                        current_state = v.copy().astype(np.clongdouble)
                        current_state = ((current_state + current_state.conj().T)/2).astype(np.cdouble)
                        mineig = np.min(np.linalg.eigvalsh(current_state)).astype(np.clongdouble)
                        if mineig < 0:
                            current_state = 1/(1-mineig)*current_state + mineig/(mineig-1)*np.eye(current_state.shape[0])
                        current_state /= np.trace(current_state)
                        best_states[k] = current_state.astype(np.cdouble)
                        
                    best_povms = deepcopy(args["povms"])
                    # clean povms TODO finish this
                        
                    best_probability = np_prob_from_states_povms(best_states,
                                                                 best_povms,
                                                                 outcomes_per_party,
                                                                 settings_per_party,
                                                                 final_state_dims,
                                                                 final_povm_dims,
                                                                 perm_states2povms,
                                                                 perm_povms2states,
                                                                 permute_states=True
                                                                 )
                    assert np.allclose(best_probability.flatten() @ objective_as_array.flatten(), best_value), \
                        "Returned probability does not match found best value"
                break
            else:
                old_value = new_value
    if verbose:
        print("BEST VALUE: ", best_value)
    if verbose > 1:
        print(args["states"])
        print(args["povms"])
    return best_value, best_states, best_povms, best_probability
    
def seesaw_l_norm(target_probability,
           outcomes_per_party,
           settings_per_party,
           Hilbert_space_dims,
           fixed_states,
           fixed_povms,
           state_support,
           povms_support,
           nof_iterations=10):

    parties = list(outcomes_per_party.keys())
    outcome_cards = list(outcomes_per_party.values())
    setting_cards = list(settings_per_party.values())

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
        # for x in _list_of_x:
        #     cvxpy_variables += [(party, x)]

    # Generate random sample states and povms and return the best value
    best_value = 1e100
    for iter_nr in range(1, nof_iterations):
        # print(f"ITERATION {i}")
        old_value = 1e100
        for s in fixed_states:
            if fixed_states[s] is None:
                seesaw_states[s] = generate_random_mixed_state(state_dims[s])
        for p in fixed_povms:
            for j, x in enumerate(fixed_povms[p]):
                if x is None:
                    seesaw_povms[p][j] = generate_random_povm(povm_dims[p],
                                                            outcomes_per_party[p])

        # Iteratively optimize until objective value converges
        for v in itertools.cycle(cvxpy_variables):
            constraints = []
            if isinstance(v, str):
                seesaw_states[v] = cp.Variable(shape=(state_dims[v],)*2,
                                               hermitian=True)
                constraints += [seesaw_states[v] >> 0,
                               cp.trace(seesaw_states[v]) == 1]
            else:
                for x in v[1:]:
                    seesaw_povms[v[0]][x] = [cp.Variable(shape=(povm_dims[v[0]],)*2,
                                                      hermitian=True)
                                          for _ in range(outcomes_per_party[v[0]])]
                    constraints += [sum(seesaw_povms[v[0]][x]) == np.eye(povm_dims[v[0]]),
                                    *[e >> 0 for e in seesaw_povms[v[0]][x]]]
            
            p = np.zeros((*outcome_cards, *setting_cards), dtype=object)
            for outs in np.ndindex(*outcome_cards):
                for ins in np.ndindex(*setting_cards):
                    list_of_mmnts = [seesaw_povms[p][ins[i]][outs[i]] 
                                      for i, p in enumerate(parties)]
                    mmnt = list_of_mmnts[0]
                    for i in range(1, len(list_of_mmnts)):
                        mmnt = cp.kron(mmnt, list_of_mmnts[i])
                        
                    list_of_states = list(seesaw_states.values())
                    state = list_of_states[0]
                    for i in range(1, len(list_of_states)):
                        state = cp.kron(state, list_of_states[i])
                    
                    p[(*outs, *ins)] = cp.real(cp.trace(state @ mmnt))
            
            l1_norm = cp.Variable(nonneg=True)
            
            for outs in np.ndindex(*outcome_cards):
                for ins in np.ndindex(*setting_cards):
                    # Pedro! Here you should make sure the objective only cares
                    # about probability terms that are valid
                    # for the 3 party scenario:
                    a, b, c = outs
                    x, y, z = ins
                    if (x == b) and (z == b):
                        expr = p[(*outs, *ins)] - target_probability[(*outs, *ins)]
                        constraints += [ -l1_norm <= expr ] + [ expr <= l1_norm]
                    
            problem = cp.Problem(cp.Minimize(l1_norm), constraints)
            
            problem.solve(verbose=False)
            
            if problem.status == 'optimal':
                new_value = l1_norm.value
                print(f"\nIteration: {iter_nr}. |p-p_target|_1 = {l1_norm.value}", end="")
                
                if isinstance(v, str):
                    seesaw_states[v] = seesaw_states[v].value
                elif isinstance(v, tuple):
                    for x in v[1:]:
                        as_values = [e.value for e in seesaw_povms[v[0]][x]]
                        seesaw_povms[v[0]][x] = as_values.copy()
            else:
                print(f"Iteration: {iter_nr} Something went wrong solving the SDP.")
                
        
            if abs(new_value - old_value) < 1e-7:
                print(f"   Converged!", end="")
                # print(seesaw_states)
                # print(seesaw_povms)
                best_value = new_value if new_value < best_value \
                    else best_value
                print(" Best: {best_value}", end="")
                break
            else:
                old_value = new_value
                
            #### WHEN TO STOP
            
            if abs(best_value) < 1e-7:
                print("BEST VALUE: ", best_value, seesaw_states, seesaw_povms)
                return best_value, seesaw_states, seesaw_povms
    return -1
                
def seesaw_with_broadcasting(
           party2broadcast_cards,
           objective_as_array_fine_grained,
           Hilbert_space_dims,
           fixed_states,
           fixed_povms,
           state_support,
           povms_support,
           seesaw_max_iterations=10,
           verbose=1):
    parties = list(party2broadcast_cards.keys())
    outcomes_per_party = {}
    settings_per_party = {}
    for p, k in party2broadcast_cards.items():
        outcomes_per_party[p] = np.prod(k[0])
        settings_per_party[p] = np.prod(k[1])
    party2idx = {p: i for i, p in enumerate(parties)}
    outcome_cards = list(outcomes_per_party.values())
    setting_cards = list(settings_per_party.values())

    # If the objective is specified for a finegrained scenario, i.e., if in the
    # bell scenario we broadcast Bob to 2 Bobs, AB->AB1B2, with cardinalities (222|222),
    # then the shape of the objective is [2, 2, 2, 2, 2, 2]. For compatibility
    # with previous code, we reshape it as [2, 4, 2, 4] and impose constraints
    # on the 4 outcome-4 setting POVM of the unified party B=(B1,B2) to
    # simulate the channel + 2 local measurements for Bob1 and Bob2.
    objective_as_array = objective_as_array_fine_grained.reshape([*outcome_cards, *setting_cards])

    povm_dims = {p: np.prod([Hilbert_space_dims[h] for h in sup]) for p, sup
                 in povms_support.items()}
    state_dims = {s: np.prod([Hilbert_space_dims[h] for h in sup]) for s, sup
                  in state_support.items()}
    all_state_supports = flatten(list(state_support.values()))
    all_povm_supports = flatten(list(povms_support.values()))    
    final_state_dims = [Hilbert_space_dims[s] for s in all_state_supports]
    final_povm_dims = [Hilbert_space_dims[s] for s in all_povm_supports]
    perm_povms2states = find_permutation(all_state_supports, all_povm_supports)
    perm_states2povms = find_permutation(all_povm_supports, all_state_supports)

    seesaw_states = deepcopy(fixed_states)
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
                    party2idx[party]], "Number of outcomes is not correct"
                assert seesaw_povms[party][x][0].shape == (
                    povm_dims[party], povm_dims[party]), \
                    f"Dimension of fixed POVM {party} does not match that deduced from Hilbert_space_dims"
        # cvxpy_variables += [(party, *_list_of_x)]
        for x in _list_of_x:
            cvxpy_variables += [(party, x)]


    # Set up problems and their parameters for each unknown value
    cvxpy_probs = {}
    cvxpy_parameters = {}
    cvxpy_name2variable = {}
    for variable in cvxpy_variables:
        if isinstance(variable, str):
            # Set up problem for unknown states
            state_dim = (state_dims[variable],) * 2
            BellOp = cp.Parameter(shape=state_dim, hermitian=True)
            rho = cp.Variable(shape=state_dim, hermitian=True, name=str(variable)) \
                if seesaw_states[variable] is None else seesaw_states[variable]
            cvxpy_name2variable[rho.name()] = rho
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
            op_dims = (outcomes_per_party[party], settings_per_party[party], )
            povm_dim = povm_dims[party]
            BellOps = np.zeros(op_dims, dtype=object)
            vars_in_prob = np.zeros(op_dims, dtype=object)
            for a, x in np.ndindex(*op_dims):
                BellOps[a, x] = cp.Parameter(shape=(povm_dim,) * 2,
                                             hermitian=True)
                if fixed_povms[party][x] is None:
                    vars_in_prob[a, x] = cp.Variable(shape=(povm_dim,) * 2,
                                                    hermitian=True,
                                                    name=str(variable)+'_'+str(a)+str(x))
                    cvxpy_name2variable[vars_in_prob[a, x].name()] = \
                        vars_in_prob[a, x]
                else:
                    vars_in_prob[a, x] = seesaw_povms[party][x][a]
                    
            # constraints = []
            # for row in vars_in_prob:
            #     summ = row.sum()
            #     if not isinstance(summ, np.ndarray):
            #         constraints.append(summ == np.eye(povm_dim))
            #     for v in row:
            #         if not isinstance(v, np.ndarray):
            #             constraints.append(v >> 0)
            constraints = []
            if party == 'A':
                for x in range(3):
                    summ = 0
                    for a in range(2):
                        if not isinstance(vars_in_prob[a, x], np.ndarray):
                            constraints.append(vars_in_prob[a, x] >> 0)
                        summ += vars_in_prob[a, x]
                    if not isinstance(summ, np.ndarray):
                        constraints.append(summ == np.eye(povm_dim))
            elif party == 'B':
                for y in range(4):
                    for b in range(4):
                        constraints += [vars_in_prob[b, y] >> 0]
                    constraints += [vars_in_prob[:, y].sum() == np.eye(povm_dim)]
                for b2 in range(2):
                    for y2 in range(2):
                        y1_0 = 0
                        summ_0 = 0
                        for b1 in range(2):
                            summ_0 += vars_in_prob[np.ravel_multi_index([b1, b2], dims=(2,2)),
                                                   np.ravel_multi_index([y1_0, y2], dims=(2,2))]
                        y1_1 = 1
                        summ_1 = 0
                        for b1 in range(2):
                            summ_1 += vars_in_prob[np.ravel_multi_index([b1, b2], dims=(2,2)),
                                                   np.ravel_multi_index([y1_1, y2], dims=(2,2))]
                        constraints += [summ_0 == summ_1]
                for b1 in range(2):
                    for y1 in range(2):
                        y2_0 = 0
                        summ_0 = 0
                        for b2 in range(2):
                            summ_0 += vars_in_prob[np.ravel_multi_index([b1, b2], dims=(2,2)),
                                                   np.ravel_multi_index([y1, y2_0], dims=(2,2)),]
                        y2_1 = 1
                        summ_1 = 0
                        for b2 in range(2):
                            summ_1 += vars_in_prob[np.ravel_multi_index([b1, b2], dims=(2,2)),
                                                   np.ravel_multi_index([y1, y2_1], dims=(2,2)),
                                                   ]
                        constraints += [summ_0 == summ_1]
            else:
                raise ValueError("Party not recognised")
            
            objective = cp.real(cp.trace(sum([BellOps[a, x] @ vars_in_prob[a, x]
                                              for a, x in
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
    for i in range(1, seesaw_max_iterations):
        print(f"ITERATION {i}. Choosing random initial states and POVMs.")
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
        LOOP_LIMIT = 100
        for loop_index in range(LOOP_LIMIT):
            # for v in itertools.cycle(cvxpy_variables):
            for v in cvxpy_variables:
                BellOps = cvxpy_parameters[v]
                prob = cvxpy_probs[v]
                if isinstance(v, str):
                    # Optimizing state
                    BellOps.value = compute_effective_Bell_operator(
                        **args, variable_to_optimise_over=v)
                    prob.solve(verbose=False)
                    args["states"][v] = cvxpy_name2variable[v].value # prob.variables()[0].value
                else:
                    # Optimizing povms
                    BellOps_values = compute_effective_Bell_operator(
                        **args, variable_to_optimise_over=(v[0],*range(settings_per_party[v[0]])))
                    for x, a in np.ndindex(settings_per_party[v[0]],
                                        outcomes_per_party[v[0]]):
                        BellOps[a, x].value = BellOps_values[a, x]
                    prob.solve(verbose=False)
                    # vars_in_prob = prob.variables()
                    for x in v[1:]:
                        args["povms"][v[0]][x] = [cvxpy_name2variable[str(v)+'_'+str(a)+str(x)].value for a in
                                                range(outcomes_per_party[v[0]])]
                        # args["povms"][v[0]][x] = [Pa.value for Pa in
                        #                           vars_in_prob[:settings_per_party[v[0]]]]
                        # vars_in_prob = vars_in_prob[settings_per_party[v[0]]:]
                new_value = prob.value
                if verbose > 1:
                    if new_value < old_value:
                        print(f"WARNING: VALUE DECREASED by {abs(old_value-new_value)}")
                
            if verbose:
                print(f"Sweep {loop_index}/{LOOP_LIMIT}: {prob.value}")

            if abs(new_value - old_value) < 1e-7:
                if verbose:
                    print("CONVERGENCE")
                if new_value > best_value:
                    best_value = new_value
                    best_states = deepcopy(args["states"])
                    # clean states
                    for k, v in best_states.items():
                        current_state = v.copy().astype(np.clongdouble)
                        current_state = ((current_state + current_state.conj().T)/2).astype(np.cdouble)
                        mineig = np.min(np.linalg.eigvalsh(current_state)).astype(np.clongdouble)
                        if mineig < 0:
                            current_state = 1/(1-mineig)*current_state + mineig/(mineig-1)*np.eye(current_state.shape[0])
                        current_state /= np.trace(current_state)
                        best_states[k] = current_state.astype(np.cdouble)
                        
                    best_povms = deepcopy(args["povms"])
                    # clean povms TODO finish this
                        
                    best_probability = np_prob_from_states_povms(best_states,
                                                                 best_povms,
                                                                 outcomes_per_party,
                                                                 settings_per_party,
                                                                 final_state_dims,
                                                                 final_povm_dims,
                                                                 perm_states2povms,
                                                                 perm_povms2states,
                                                                 permute_states=True
                                                                 )
                    assert np.allclose(best_probability.flatten() @ objective_as_array.flatten(), best_value), \
                        "Returned probability does not match found best value"
                break
            else:
                old_value = new_value
    if verbose:
        print("BEST VALUE: ", best_value)
    if verbose > 1:
        print(args["states"])
        print(args["povms"])
    return best_value, best_states, best_povms, best_probability
       
    
    
    
if __name__ == '__main__':
    ############ l norm
    
    # dag_global = {'psiAB': ['A', 'B'],
    #               'psiBC': ['B', 'C']}
    # outcomes_per_party = {'A': 2, 'B': 2, 'C': 2}
    # settings_per_party = {'A': 2, 'B': 1, 'C': 2}

    # scenario = NetworkScenario(dag_global, outcomes_per_party, settings_per_party)

    # ops = generate_ops(outcomes_per_party, settings_per_party)
    # A = [1 - 2*ops[0][x][0] for x in range(settings_per_party['A'])]
    # B = [1 - 2*ops[1][x][0] for x in range(settings_per_party['B'])]
    # C = [1 - 2*ops[2][x][0] for x in range(settings_per_party['C'])]

    # # Fix local dimensions
    # LOCAL_DIM = 2
    # Hilbert_space_dims = {H: LOCAL_DIM for H in scenario.Hilbert_spaces}

    # state_support = {'psiAB': ['H_A_psiAB', 'H_B_psiAB'],
    #                  'psiBC': ['H_B_psiBC', 'H_C_psiBC']}
    # povms_support = {'A': ['H_A_psiAB'],
    #                  'B': ['H_B_psiAB', 'H_B_psiBC'],
    #                  'C': ['H_C_psiBC']}

    # import qutip as qt
    # fixed_states = {'psiAB': qt.rand_dm(4).data.A,
    #                 'psiBC': qt.rand_dm(4).data.A}
    # fixed_measurements = {'A': [None, None],
    #                       'B': [None],
    #                       'C': [None, None]}

    # seesaw_l_norm(target_probability=1/8*np.ones((2, 2, 2, 2, 1, 2)),
    #        outcomes_per_party=outcomes_per_party,
    #        settings_per_party=settings_per_party,
    #        Hilbert_space_dims=Hilbert_space_dims,
    #        fixed_states=fixed_states,
    #        fixed_povms=fixed_measurements,
    #        state_support=state_support,
    #        povms_support=povms_support,
    #        nof_iterations=100)

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

    # bell_state = np.expand_dims(np.array([1, 0, 0, 1]), axis=1)/np.sqrt(2)
    # bell_state = bell_state @ bell_state.T.conj()
    # A0 = [state.proj().data.A for state in qt.sigmaz().eigenstates()[1]]
    # A1 = [state.proj().data.A for state in qt.sigmax().eigenstates()[1]]
    # B0 = [state.proj().data.A for state in (qt.sigmaz()+qt.sigmax()).eigenstates()[1]]
    # B1 = [state.proj().data.A for state in (qt.sigmaz()-qt.sigmax()).eigenstates()[1]]
    # _A = [A0, A1]
    # _B = [B0, B1]

    # seesaw(outcomes_per_party=outcomes_per_party,
    #        settings_per_party=settings_per_party,
    #        objective_as_array=CHSH_array,
    #        Hilbert_space_dims=Hilbert_space_dims,
    #        fixed_states={'psiAB': None},
    #        fixed_povms={'A': [None, None], 'B': [None, None]},
    #        state_support=state_support,
    #        povms_support=povms_support)
    
    # fixed_states = {'psiAB': bell_state}
    # fixed_measurements = {'A': [A0, A1],
    #                       'B': [B0, B1]}
    
    # final_state_dims = [Hilbert_space_dims[s] for s in flatten(list(state_support.values()))]
    # final_povm_dims = [Hilbert_space_dims[s] for s in flatten(list(povms_support.values()))]
    # perm_povms2states = find_permutation(flatten(list(state_support.values())), flatten(list(povms_support.values())))
    # perm_states2povms = find_permutation(flatten(list(povms_support.values())), flatten(list(state_support.values())))
    # p = np_prob_from_states_povms(fixed_states, fixed_measurements, outcomes_per_party, settings_per_party,
    #                               final_state_dims, final_povm_dims, perm_states2povms, perm_povms2states, permute_states=False)
    
    
    # assert abs(p.flatten().T @ CHSH_array.flatten() - 2*np.sqrt(2))<1e-7, "2sqrt(2) is not achieved, initial mmnts are not good"
    
    # ########## Broadcasting

    
    outcomes_per_party = {'A': 2, 'B1': 2, 'B2': 2}
    settings_per_party = {'A': 3, 'B1': 2, 'B2': 2}
    ops = generate_ops(outcomes_per_party, settings_per_party)
    A = [1 - 2*ops[0][x][0] for x in range(settings_per_party['A'])]
    B1 = [1 - 2*ops[1][x][0] for x in range(settings_per_party['B1'])]
    B2 = [1 - 2*ops[2][x][0] for x in range(settings_per_party['B2'])]
    CHSH = (A[0] - A[1])*B2[0] + (A[0] + A[1])*B2[1] 
    BROADCAST_CHSH = CHSH*(B1[0] + B1[1]) + 2*A[2]*(B1[1] - B1[0])
    BROADCAST_CHSH_array = Bell_CG2prob(BROADCAST_CHSH, outcomes_per_party, settings_per_party)

    # Fix local dimensions
    dag = {'psiAB': ['A', 'B']}
    scenario = NetworkScenario(dag, {'A': 2, 'B': 2}, {'A': 2, 'B': 2})  # Here the outcomes are trash
    LOCAL_DIM = 2
    Hilbert_space_dims = {H: LOCAL_DIM for H in scenario.Hilbert_spaces}

    state_support = {'psiAB': ['H_A_psiAB', 'H_B_psiAB']}
    povms_support = {'A': ['H_A_psiAB'], 'B': ['H_B_psiAB']}

    bell_state = np.expand_dims(np.array([1, 0, 0, 1]), axis=1)/np.sqrt(2)
    bell_state = bell_state @ bell_state.T.conj()
    A0 = [state.proj().data.A for state in qt.sigmaz().eigenstates()[1]]
    A1 = [state.proj().data.A for state in qt.sigmax().eigenstates()[1]]
    B0 = [state.proj().data.A for state in (qt.sigmaz()+qt.sigmax()).eigenstates()[1]]
    B1 = [state.proj().data.A for state in (qt.sigmaz()-qt.sigmax()).eigenstates()[1]]
    _A = [A0, A1]
    _B = [B0, B1]

    seesaw_with_broadcasting(
           party2broadcast_cards = {'A': (2, 3), 'B': [(4), (4)]},
           objective_as_array_fine_grained=BROADCAST_CHSH_array,
           Hilbert_space_dims=Hilbert_space_dims,
           fixed_states={'psiAB': None},
           fixed_povms={'A': [None, None, None], 'B': [None, None, None, None]},
           state_support=state_support,
           povms_support=povms_support)
    
    fixed_states = {'psiAB': bell_state}
    fixed_measurements = {'A': [A0, A1],
                          'B': [B0, B1]}
    
    final_state_dims = [Hilbert_space_dims[s] for s in flatten(list(state_support.values()))]
    final_povm_dims = [Hilbert_space_dims[s] for s in flatten(list(povms_support.values()))]
    perm_povms2states = find_permutation(flatten(list(state_support.values())), flatten(list(povms_support.values())))
    perm_states2povms = find_permutation(flatten(list(povms_support.values())), flatten(list(state_support.values())))
    p = np_prob_from_states_povms(fixed_states, fixed_measurements, outcomes_per_party, settings_per_party,
                                  final_state_dims, final_povm_dims, perm_states2povms, perm_povms2states, permute_states=False)
    
    
    assert abs(p.flatten().T @ CHSH_array.flatten() - 2*np.sqrt(2))<1e-7, "2sqrt(2) is not achieved, initial mmnts are not good"
    
    
    
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
    
    # value = seesaw(outcomes_per_party=outcomes_per_party,
    #        settings_per_party=settings_per_party,
    #        objective_as_array=MERMIN_as_array,
    #        Hilbert_space_dims=Hilbert_space_dims,
    #        fixed_states=fixed_states,
    #        fixed_povms=fixed_measurements,
    #        state_support=state_support,
    #        povms_support=povms_support)

    # ############################# MERMIN Global ##############################

    # dag_global = {'psiABC': ['A', 'B', 'C']}
    # outcomes_per_party = {'A': 2, 'B': 2, 'C': 2}
    # settings_per_party = {'A': 2, 'B': 2, 'C': 2}

    # scenario = NetworkScenario(dag_global, outcomes_per_party, settings_per_party)

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

    # state_support = {'psiABC': ['H_A_psiABC', 'H_B_psiABC', 'H_C_psiABC']}
    # povms_support = {'A': ['H_A_psiABC'],
    #                  'B': ['H_B_psiABC'],
    #                  'C': ['H_C_psiABC']}

    # import qutip as qt
    # fixed_states = {'psiABC': None}#qt.rand_dm(8).data.A}
    # fixed_measurements = {'A': [None, None], 'B': [None, None], 'C': [None, None]}

    # seesaw(outcomes_per_party=outcomes_per_party,
    #        settings_per_party=settings_per_party,
    #        objective_as_array=MERMIN_as_array,
    #        Hilbert_space_dims=Hilbert_space_dims,
    #        fixed_states=fixed_states,
    #        fixed_povms=fixed_measurements,
    #        state_support=state_support,
    #        povms_support=povms_support)
    
    
