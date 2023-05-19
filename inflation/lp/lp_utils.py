# Send problems to LP solvers

import numpy as np

from typing import List, Dict, Any
from copy import deepcopy
from mosek.fusion import *


def solveLP_MosekFUSION(objective: Dict = None,
                        known_vars: Dict = None,
                        inequalities: List[Dict] = None,
                        equalities: List[Dict] = None
                        ) -> Any:
    # Internal function to solve the LP using Mosek FUSION API
    # Return objective value

    # if objective is None:
    #     var_objective = {}
    # else:
    #     var_objective = objective.copy()
    # if known_vars is None:
    #     known_vars = {}
    # if inequalities is None:
    #     var_inequalities = []
    # else:
    #     var_inequalities = deepcopy(inequalities)
    # if equalities is None:
    #     var_equalities = []
    # else:
    #     var_equalities = deepcopy(equalities)

    # Define variables for LP
    variables = set()
    variables.update(objective.keys())
    for ineq in inequalities:
        variables.update(ineq.keys())
    for eq in equalities:
        variables.update(eq.keys())
    variables.difference_update(known_vars.keys())

    # Compute c0, the constant part of objective
    c0 = 0
    for x in objective.keys():
        if x in known_vars.keys():
            c0 += objective[x] * known_vars[x]

    # Create dictionary var_index - monomial : index
    var_index = {x : i for i, x in enumerate(variables)}

    # Create matrices A, C and vectors b, d such that Ax + b >= 0, Cx + d == 0
    A = np.zeros((len(inequalities), len(variables)))
    b = np.zeros(len(inequalities))
    for i, inequality in enumerate(inequalities):
        vars = set(inequality)
        for x in vars.difference(set(known_vars)):
            A[i, var_index[x]] = inequality[x] # Fills A with coefficients of vars, excluding known vars
        for x in vars:
            if x in known_vars.keys():
                b[i] += inequality[x] * known_vars[x] # Fills b with constant values
    C = np.zeros((len(equalities), len(variables)))
    d = np.zeros(len(equalities))
    for i, equality in enumerate(equalities):
        vars = set(equality)
        for x in vars.difference(set(known_vars)):
            C[i, var_index[x]] = equality[x]
        for x in vars:
            if x in known_vars.keys():
                d[i] += equality[x] * known_vars[x]

    with Model("LP") as M:
        # Set up the problem as a primal LP

        # Define variables
        x = M.variable("x", len(variables), Domain.greaterThan(0.0))

        # Define constraints
        for i in range(len(inequalities)):
            M.constraint("c" + str(i), Expr.add(Expr.dot(A[i], x), b[i]), Domain.greaterThan(0))

        for i in range(len(equalities)):
            M.constraint("c" + str(len(inequalities) + i), Expr.add(Expr.dot(C[i], x), d[i]), Domain.equalsTo(0))

        # Define objective
        obj = c0
        for var in set(objective).difference(set(known_vars)):
            obj = Expr.add(obj, Expr.mul(x.index(var_index[var]), objective[var]))
        M.objective(ObjectiveSense.Maximize, obj)

        M.solve()
        print(M.primalObjValue())






























