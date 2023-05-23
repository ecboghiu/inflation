import numpy as np

from typing import List, Dict
from mosek.fusion import *


def solveLP_MosekFUSION(objective: Dict = None,
                        known_vars: Dict = None,
                        inequalities: List[Dict] = None,
                        equalities: List[Dict] = None
                        ) -> Dict:
    """Internal function to solve an LP with the Mosek FUSION API.

    Parameters
    ----------
    objective : dict
        Monomials (keys) and coefficients (values) that describe
        the objective function
    known_vars : dict
        Monomials (keys) and known values of the monomials
    inequalities: list of dict
        Inequality constraints with monomials (keys) and coefficients (values)
    equalities: list of dict
        Equality constraints with monomials (keys) and coefficients (values)

    Returns
    -------
    dict
        Primal objective value, dual objective value, problem status, dual
        certificate, x values

    """

    # Define variables for LP, excluding those with known values
    variables = set()
    variables.update(objective.keys())
    for ineq in inequalities:
        variables.update(ineq.keys())
    for eq in equalities:
        variables.update(eq.keys())
    variables.difference_update(known_vars.keys())

    # Compute c0, the constant term in the objective function
    c0 = 0
    for x in objective.keys():
        if x in known_vars.keys():
            c0 += objective[x] * known_vars[x]

    # Create dictionary var_index - monomial : index
    var_index = {x: i for i, x in enumerate(variables)}

    # Create matrix A, vector b such that Ax + b >= 0
    A = np.zeros((len(inequalities), len(variables)))
    b = np.zeros(len(inequalities))
    for i, inequality in enumerate(inequalities):
        monomials = set(inequality.keys())
        for x in monomials.difference(known_vars.keys()):
            A[i, var_index[x]] = inequality[x]
        for x in monomials:
            if x in known_vars.keys():
                b[i] += inequality[x] * known_vars[x]

    # Create matrix C, vector d such that Cx + d == 0
    C = np.zeros((len(equalities), len(variables)))
    d = np.zeros(len(equalities))
    for i, equality in enumerate(equalities):
        monomials = set(equality)
        for x in monomials.difference(known_vars.keys()):
            C[i, var_index[x]] = equality[x]
        for x in monomials:
            if x in known_vars.keys():
                d[i] += equality[x] * known_vars[x]

    with Model("LP") as M:
        # Set up the problem as a primal LP

        # Define variables
        x = M.variable("x", len(variables), Domain.greaterThan(0.0))

        # Define constraints
        for i in range(len(inequalities)):
            M.constraint("ineq" + str(i), Expr.add(Expr.dot(A[i], x), b[i]),
                         Domain.greaterThan(0))
        for i in range(len(equalities)):
            M.constraint("eq" + str(i), Expr.add(Expr.dot(C[i], x), d[i]),
                         Domain.equalsTo(0))

        # Define objective function
        obj = c0
        for var in set(objective.keys()).difference(known_vars.keys())):
            obj = Expr.add(obj, Expr.mul(x.index(var_index[var]),
                                         objective[var]))
        M.objective(ObjectiveSense.Maximize, obj)

        # Solve the LP
        M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        M.solve()

        x_values = dict(zip(variables, x.level()))

        status = M.getProblemStatus()
        if status == ProblemStatus.PrimalAndDualFeasible:
            status_str = "feasible"
            primal = M.primalObjValue()
            dual = M.dualObjValue()
        else:
            status_str = "infeasible"

        # Derive certificate here

        return {
            "primal_value": primal,
            "dual_value": dual,
            "status": status_str,
            "dual_certificate": None,
            "x": x_values
        }
