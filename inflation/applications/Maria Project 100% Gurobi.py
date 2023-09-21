import gurobipy as gp
from gurobipy import GRB
from itertools import product, repeat
import numpy as np

gurobi_status_codes = [
    'LOADED',
    'OPTIMAL',
    'INFEASIBLE',
    'INF_OR_UNBD',
    'UNBOUNDED',
    'CUTOFF',
    'ITERATION_LIMIT',
    'NODE_LIMIT',
    'TIME_LIMIT',
    'SOLUTION_LIMIT',
    'INTERRUPTED',
    'NUMERIC',
    'SUBOPTIMAL',
    'INPROGRESS',
    'USER_OBJ_LIMIT',
    'WORK_LIMIT']
time_limit = GRB.INFINITY
tol = 1e-4
return_dist = True
print_model = True

cardA = 2
cardB = 2
cardC = 2
cardX = 2
cardZ = 2
rA = tuple(range(cardA))
rB = tuple(range(cardB))
rC = tuple(range(cardC))
rX = tuple(range(cardX))
rZ = tuple(range(cardZ))

def PR_box_with_visibility(a, b, x, y, v):
    return (1 + v * (-1) ** (a + b + x * y)) / 4
def Charlie_as_Setting_For_Bob_Box(v=np.sqrt(2)/2):
    tripartite_box = np.zeros((2, 2, 2, 2, 2), dtype=object)
    for (x, z, a, b, c) in np.ndindex(tripartite_box.shape):
        tripartite_box[a, b, c, x, z] = 1 / 2 * PR_box_with_visibility(a, b, x, c, v)
    return tripartite_box

def Postselection_Box(v=np.sqrt(2)/2, post_selection_probability=1/4):
    tripartite_box = np.zeros((2, 2, 2, 2, 2), dtype=object)
    # white_noise_box = np.ones((2, 2, 2, 2), dtype=object) / 4
    for (x, z, a, c) in np.ndindex((2, 2, 2, 2)):
        tripartite_box[
            a, 0, c, x, z] = post_selection_probability * PR_box_with_visibility(
            a, c, x, z, v)
    tripartite_box[:, 1] = 1/4 - tripartite_box[:, 0]
    return tripartite_box

def NSBoxFrom1to2(w: float=1, v=np.sqrt(2)/2, post_selection_probability=1/4):
    return (1-w)*Charlie_as_Setting_For_Bob_Box(v) + w*Postselection_Box(v, post_selection_probability=post_selection_probability)

quantum_box = NSBoxFrom1to2(w=1/2, v=np.sqrt(2)/2, post_selection_probability=1/4)

with gp.Env(empty=True) as env:
    # env.setParam('OutputFlag', 0) # To supress output
    env.start()
    with gp.Model("qcp", env=env) as m:
        m.params.NonConvex = 2  # Using quadratic equality constraints.
        m.setParam('FeasibilityTol', tol)
        m.setParam('OptimalityTol', 0.01)
        m.setParam('TimeLimit', time_limit)
        m.setParam('Presolve', 0)
        m.setParam('PreSparsify', 1)
        m.setParam('PreQLinearize', 1)
        # m.setParam('Method', 2)
        # m.setParam('PreMIQCPForm', 2)
        m.setParam('PreDepRow', 1)
        m.setParam('Symmetry', 2)
        m.setParam('Heuristics', 1.0)
        m.setParam('RINS', 0)
        m.setParam('MIPFocus', 3)
        m.setParam('MinRelNodes', 0)
        m.setParam('ZeroObjNodes', 0)
        m.setParam('ImproveStartGap', 0)

        # Placeholder observable probabilities
        bilocal_conditional_probs = m.addMVar((cardA, cardB, cardC, cardX, cardZ), lb=0, ub=1, name="P_bilocal")
        in_hull_conditional_probs = m.addMVar((cardA, cardB, cardC, cardX, cardZ), lb=0, ub=1, name="P_quantum")

        # Define fully unpacked probabilities
        unpacked_A_cardinalities = tuple(repeat(cardA, times=cardX))
        unpacked_C_cardinalities = tuple(repeat(cardC, times=cardZ))
        unpacked_all_cardinalities = unpacked_A_cardinalities + (cardB,) + unpacked_C_cardinalities
        unpacked_probabilities = m.addMVar(unpacked_all_cardinalities, lb=0, ub=1, name="P_unpacked")
        unpacked_AC = m.addMVar(unpacked_A_cardinalities + unpacked_C_cardinalities, lb=0, ub=1,
                               name="P_AACC")
        unpacked_A = m.addMVar(unpacked_A_cardinalities, lb=0, ub=1, name="P_AA")
        unpacked_C = m.addMVar(unpacked_C_cardinalities, lb=0, ub=1, name="P_CC")

        normalization = unpacked_probabilities.sum()
        m.addConstr(unpacked_probabilities.sum() == 1, name="All Unpacked Normalization")
        unpacked_C_marginal_slice = tuple(repeat(np.s_[:], times=cardZ))
        unpacked_A_marginal_slice = tuple(repeat(np.s_[:], times=cardX))
        for responseA in np.ndindex(unpacked_A_cardinalities):
            m.addConstr(unpacked_A[responseA] == unpacked_AC[responseA + unpacked_C_marginal_slice].sum())
        for responseC in np.ndindex(unpacked_C_cardinalities):
            m.addConstr(unpacked_C[responseC] == unpacked_AC[unpacked_A_marginal_slice + responseC].sum())
        one_party_marginal_slice = np.index_exp[:]
        for (responseA, responseC) in product(np.ndindex(unpacked_A_cardinalities), np.ndindex(unpacked_C_cardinalities)):
            AB_marginal_slice = np.s_[responseA + np.index_exp[:] + responseC]
            m.addConstr(unpacked_AC[responseA + responseC] == unpacked_probabilities[AB_marginal_slice].sum())
            m.addConstr(unpacked_AC[responseA + responseC] == unpacked_A[responseA] * unpacked_C[responseC])

        # Relate observable probabilities to unpacked probabilities
        for (a, b, c, x, z) in product(rA, rB, rC, rX, rZ):
            Alice_indices = list(unpacked_A_marginal_slice)
            Alice_indices[x] = a
            Charlie_indices = list(unpacked_C_marginal_slice)
            Charlie_indices[z] = c
            marginal_slice = np.s_[tuple(Alice_indices) + np.index_exp[b] + tuple(Charlie_indices)]
            m.addConstr(bilocal_conditional_probs[a, b, c, x, z] == unpacked_probabilities[marginal_slice].sum())

        # # Setting the objective to be min distance from some quantum box
        # objective_terms = (bilocal_conditional_probs - quantum_box) * (bilocal_conditional_probs - quantum_box)
        # objective = objective_terms.sum()
        # m.setObjective(objective, sense=GRB.MINIMIZE)

        # Setting the objective to be max visibility
        vis = m.addVar(lb=0, ub=1, name="visibility")
        m.setObjective(vis, sense=GRB.MAXIMIZE)
        m.addConstr(
            bilocal_conditional_probs == NSBoxFrom1to2(w=1/2, v=vis, post_selection_probability=1/4))



        m.update()
        current_status = m.getAttr("Status")
        # if current_status == 1:
        #     m.presolve()
        #     current_status = m.getAttr("Status")
        # print(m.display())
        if current_status == 1:
            m.optimize()
            # current_status = ast.literal_eval(m.getJSONSolution())["SolutionInfo"]["Status"]
            current_status = m.getAttr("Status")
            current_status_string = gurobi_status_codes[current_status-1]
            print(f"Status: {current_status_string}")
        # m.printQuality()

        if m.getAttr("SolCount"):
            # (current_status in {2, 7, 8, 9, 10, 11, 12, 13, 15, 16}):
            record_to_preserve = dict()
            for var in m.getVars():
                record_to_preserve[var.VarName] = var.X
                if var.X >= 0.0001:
                    if print_model:
                        print(var.VarName, " := ", var.X)
        m.dispose()
    env.dispose()
