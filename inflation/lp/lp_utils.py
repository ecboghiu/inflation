import numpy as np

from typing import List, Dict
from mosek.fusion import Matrix, Model, ObjectiveSense, Expr, Domain, \
    OptimizeError, SolutionError, AccSolutionStatus, ProblemStatus
from scipy.sparse import dok_matrix, vstack


def solveLP_MosekFUSION(objective: Dict = None,
                        known_vars: Dict = None,
                        semiknown_vars: Dict = None,
                        inequalities: List[Dict] = None,
                        equalities: List[Dict] = None,
                        solve_dual: bool = True
                        ) -> Dict:
    """Internal function to solve an LP with the Mosek FUSION API.

    Parameters
    ----------
    objective : dict
        Monomials (keys) and coefficients (values) that describe
        the objective function
    known_vars : dict
        Monomials (keys) and known values of the monomials
    semiknown_vars : dict
        Encodes proportionality constraints between monomials
    inequalities : list of dict
        Inequality constraints with monomials (keys) and coefficients (values)
    equalities : list of dict
        Equality constraints with monomials (keys) and coefficients (values)4
    solve_dual : bool
        Whether to solve the dual (True) or primal (False) formulation

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
    if inequalities is None:
        inequalities = []
    if equalities is None:
        equalities = []

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

    with Model("LP") as M:
        if solve_dual:
            # Define dual variables:
            nof_dual_vars = nof_inequalities + nof_equalities
            y = M.variable("y", nof_dual_vars)

            # Non-positivity constraint for y_i corresponding to inequalities
            for i in range(nof_inequalities):
                M.constraint(y.index(i), Domain.lessThan(0.0))

            # Define v as vector of coefficients of the objective, v·x
            v = dok_matrix((nof_variables, 1))
            for i, x in enumerate(variables):
                try:
                    v[var_index[x], 0] = objective[x]
                except KeyError:
                    v[var_index[x], 0] = 0
            v_mosek = Matrix.sparse(*v.shape,
                                    *v.nonzero(),
                                    v[v.nonzero()].A[0])
            del v

            # Define dual constraints:
            # For primal objective function v·x, the dual constraints are
            # (A \\ C)^T·y - v == 0
            if inequalities and equalities:
                s = vstack([A, C], 'dok')
                s_mosek = Matrix.sparse(*s.shape,
                                        *s.nonzero(),
                                        s[s.nonzero()].A[0])
                del s
                transpose = s_mosek.transpose()
            elif inequalities:
                transpose = A_mosek.transpose()
            else:
                transpose = C_mosek.transpose()
            del A, C

            c = M.constraint("c", Expr.sub(Expr.mul(transpose, y), v_mosek),
                             Domain.equalsTo(0.0))

            # Define dual objective:
            # Since Ax + b >= 0 and Cx + d == 0, the dual objective is
            # -(b \\ d)·y
            if inequalities and equalities:
                bd = -vstack([b, d], 'dok')
                bd_mosek = Matrix.sparse(*bd.shape,
                                         *bd.nonzero(),
                                         bd[bd.nonzero()].A[0])
                del bd
                obj = Expr.dot(bd_mosek, y)
            elif inequalities:
                obj = Expr.dot(b_mosek, Expr.mul(-1, y))
            else:
                obj = Expr.dot(d_mosek, Expr.mul(-1, y))
            del b, d

            M.objective(ObjectiveSense.Minimize, obj)
        else:
            # Define primal variables
            x = M.variable("x", nof_variables)

            # Define primal constraints
            ineq_cons = M.constraint("ineqs",
                                     Expr.add(Expr.mul(A_mosek, x), b_mosek),
                                     Domain.greaterThan(0.0))
            eq_cons = M.constraint("eqs",
                                   Expr.add(Expr.mul(C_mosek, x), d_mosek),
                                   Domain.equalsTo(0.0))

            # Define objective function
            obj = c0
            for var in set(objective.keys()).difference(known_vars.keys()):
                obj = Expr.add(obj, Expr.mul(x.index(var_index[var]),
                                             objective[var]))
            M.objective(ObjectiveSense.Maximize, obj)

        # Solve the LP
        M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        M.solve()

        if solve_dual:
            # Get primal solution value corresponding to each dual constraint
            x_values = {
                x: c.index([var_index[x], 0]).dual()[0]
                for x in variables
            }
        else:
            x_values = dict(zip(variables, x.level()))

        status = M.getProblemStatus()
        if status == ProblemStatus.PrimalAndDualFeasible:
            status_str = "feasible"
        else:
            status_str = "infeasible"
        primal = M.primalObjValue()
        dual = M.dualObjValue()

        # Extract the certificate:
        # Certificate is contained in the dual objective function b·y
        certificate = {x: 0 for x in known_vars}

        if solve_dual:
            y_values = -y.level()
        else:
            y_values = np.concatenate((-ineq_cons.dual(), -eq_cons.dual()))

        # Each monomial with known value is associated with a sum of duals
        cons = inequalities + equalities
        for i, c in enumerate(cons):
            for x in set(c).intersection(known_vars):
                certificate[x] += y_values[i] * c[x]

        # Clean entries with coefficient zero
        for x in list(certificate.keys()):
            if np.isclose(certificate[x], 0):
                del certificate[x]

        return {
            "primal_value": primal,
            "dual_value": dual,
            "status": status_str,
            "dual_certificate": certificate,
            "x": x_values
        }
