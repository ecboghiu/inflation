import numpy as np

from typing import List, Dict
from mosek.fusion import  Matrix, Model, ObjectiveSense, Expr, Domain, \
        OptimizeError, SolutionError, \
        AccSolutionStatus, ProblemStatus
from scipy.sparse import dok_matrix


def solveLP_MosekFUSION(objective: Dict = None,
                        known_vars: Dict = None,
                        inequalities: List[Dict] = None,
                        equalities: List[Dict] = None,
                        semiknown_vars: Dict = None
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
    # Deal with unsanitary input
    if known_vars is None:
        known_vars = {}
    if semiknown_vars is None:
        semiknown_vars = {}


    # Define variables for LP, excluding those with known values
    variables = set()
    variables.update(objective.keys())
    for ineq in inequalities:
        variables.update(ineq.keys())
    internal_equalities = equalities.copy()
    for x, (c, x2) in semiknown_vars.items():
        internal_equalities.append({x: 1, x2: -c})
    for eq in internal_equalities:
        variables.update(eq.keys())
    variables.difference_update(known_vars.keys())
    nof_variables = len(variables)

    # Compute c0, the constant term in the objective function
    c0 = 0
    for x in objective.keys():
        if x in known_vars.keys():
            c0 += objective[x] * known_vars[x]

    # Create dictionary var_index - monomial : index
    var_index = {x: i for i, x in enumerate(variables)}

    # Create matrix A, vector b such that Ax + b >= 0
    nof_inequalities = len(inequalities)
    A = dok_matrix((nof_inequalities, nof_variables))
    b = dok_matrix((nof_inequalities, 1))
    for i, inequality in enumerate(inequalities):
        monomials = set(inequality.keys())
        for x in monomials.difference(known_vars.keys()):
            A[i, var_index[x]] = inequality[x]
        for x in monomials:
            if x in known_vars.keys():
                b[i, 0] += inequality[x] * known_vars[x]
    b_mosek = Matrix.sparse(*b.shape,
                            *b.nonzero(),
                            b[b.nonzero()].A[0])
    A_mosek = Matrix.sparse(*A.shape,
                            *A.nonzero(),
                            A[A.nonzero()].A[0])
    del A, b

    # Create matrix C, vector d such that Cx + d == 0
    nof_equalities = len(internal_equalities)
    C = dok_matrix((nof_equalities, nof_variables))
    d = dok_matrix((nof_equalities, 1))
    for i, equality in enumerate(internal_equalities):
        monomials = set(equality)
        for x in monomials.difference(known_vars.keys()):
            C[i, var_index[x]] = equality[x]
        for x in monomials:
            if x in known_vars.keys():
                d[i, 0] += equality[x] * known_vars[x]
    d_mosek = Matrix.sparse(*d.shape,
                            *d.nonzero(),
                            d[d.nonzero()].A[0])
    C_mosek = Matrix.sparse(*C.shape,
                            *C.nonzero(),
                            C[C.nonzero()].A[0])
    del C, d

    with Model("LP") as M:
        # Set up the problem as a primal LP


        # Define variables
        x = M.variable("x", len(variables), Domain.greaterThan(0.0))

        # Define constraints
        M.constraint("ineqs", Expr.add(Expr.mul(A_mosek, x), b_mosek),
                     Domain.greaterThan(0))
        M.constraint("eqs", Expr.add(Expr.mul(C_mosek, x), d_mosek),
                     Domain.equalsTo(0))
        # for i in range(len(inequalities)):
        #     M.constraint("ineq" + str(i), Expr.add(Expr.dot(A[i].todense(), x), b[i, 0]),
        #                  Domain.greaterThan(0))
        # for i in range(len(equalities)):
        #     M.constraint("eq" + str(i), Expr.add(Expr.dot(C[i].todense(), x), d[i, 0]),
        #                  Domain.equalsTo(0))

        # Define objective function
        obj = c0
        for var in set(objective.keys()).difference(known_vars.keys()):
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
        else:
            status_str = "infeasible"
        primal = M.primalObjValue()
        dual = M.dualObjValue()

        # Derive certificate here

        return {
            "primal_value": primal,
            "dual_value": dual,
            "status": status_str,
            "dual_certificate": None,
            "x": x_values
        }
