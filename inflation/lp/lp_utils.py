import mosek
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
    variables = list(variables)
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

            # Define v as vector of coefficients of the primal objective, v·x
            v = dok_matrix((nof_variables, 1))
            for x, i in var_index.items():
                try:
                    v[i, 0] = objective[x]
                except KeyError:
                    v[i, 0] = 0
            v_mosek = Matrix.sparse(*v.shape,
                                    *v.nonzero(),
                                    v[v.nonzero()].A[0])
            del v

            # Define dual constraints:
            # For primal objective function v·x, the dual constraints are
            # (A \\ C)^T·y - v == 0
            if inequalities and internal_equalities:
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
            if inequalities and internal_equalities:
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
            x_values = dict(zip(variables, c.dual()))
        else:
            x_values = dict(zip(variables, x.level()))

        status = M.getProblemStatus()
        if status == ProblemStatus.PrimalAndDualFeasible:
            status_str = "feasible"
        else:
            status_str = "infeasible"

        if solve_dual:
            primal = M.dualObjValue() + c0
            dual = M.primalObjValue() + c0
        else:
            primal = M.primalObjValue()
            dual = M.dualObjValue()

        # Extract the certificate:
        # Certificate is contained in the dual objective function b·y
        certificate = {x: 0 for x in known_vars}

        if solve_dual:
            y_values = -y.level()
        else:
            y_values = np.hstack((-ineq_cons.dual(), -eq_cons.dual()))

        # Each monomial with known value is associated with a sum of duals
        cons = inequalities + internal_equalities
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


def solveLP_Mosek(objective: Dict = None,
                  known_vars: Dict = None,
                  semiknown_vars: Dict = None,
                  inequalities: List[Dict] = None,
                  equalities: List[Dict] = None,
                  solve_dual: bool = True
                  ) -> Dict:
    """Internal function to solve an LP with the Mosek Optimizer API.

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

    # Since the value of infinity is ignored, define it for symbolic purposes
    inf = 0.0

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
    variables = sorted(variables)

    # Create dictionary var_index - monomial : index
    var_index = {x: i for i, x in enumerate(variables)}

    with mosek.Task() as task:
        numcon = len(inequalities + internal_equalities)
        numvar = len(variables)
        nof_inequalities = len(inequalities)
        nof_equalities = len(internal_equalities)

        # Create sparse matrix A of constraints
        constraints = inequalities + internal_equalities
        A = dok_matrix((numcon, numvar))
        b = np.zeros(numcon)
        for i, cons in enumerate(constraints):
            for x in set(cons).difference(known_vars):
                A[i, var_index[x]] = cons[x]
            for x in set(cons).intersection(known_vars):
                b[i] -= cons[x] * known_vars[x]
        b = b.tolist()

        if solve_dual:
            numcon = len(variables)
            numvar = len(inequalities + internal_equalities)

            matrix = A.asformat('csr', copy=False)

            # Set bound keys and values for constraints (primal objective)
            # All equalities since primal variable x is free
            bkc = [mosek.boundkey.fx] * numcon
            blc = buc = list(objective.values())

            # Set bound keys and values for variables
            # Non-positivity for y corresponding to inequalities
            bkx = [mosek.boundkey.up] * nof_inequalities + \
                  [mosek.boundkey.fr] * nof_equalities
            blx = [-inf] * numvar
            bux = [0.0] * nof_inequalities + [+inf] * nof_equalities

            # Objective function coefficients
            c = b

            # The constant fixed term in the objective function is 0 for the
            # dual formulation
            c0 = 0

            # Set the objective sense
            task.putobjsense(mosek.objsense.minimize)


        else:
            matrix = A.asformat('csc', copy=False)

            # Set bound keys and bound values (lower and upper) for constraints
            # Ax + b >= 0 -> Ax >= -b
            bkc = [mosek.boundkey.lo] * nof_inequalities + \
                  [mosek.boundkey.fx] * nof_equalities
            blc = b
            buc = [+inf] * nof_inequalities + \
                  b[nof_inequalities:numcon] * nof_equalities

            # Set bound keys and bound values (lower and upper) for variables
            bkx = [mosek.boundkey.fr] * numvar
            blx = [-inf] * numvar
            bux = [+inf] * numvar

            # Objective function coefficients
            c = np.zeros(numvar)
            for x in set(objective).difference(known_vars):
                c[var_index[x]] = objective[x]

            # Compute c0, the constant (fixed) term in the objective function
            c0 = 0
            for x in set(objective).intersection(known_vars):
                c0 += objective[x] * known_vars[x]

            # Set the objective sense
            task.putobjsense(mosek.objsense.maximize)

        print(matrix.indptr[:-1])
        print(matrix.indptr[1:])
        print(matrix.indices)
        print(matrix.data)


        # Add all the problem data to the task
        task.inputdata(maxnumcon=numcon,
                       maxnumvar=numvar,
                       c=c,
                       cfix=c0,
                       aptrb=matrix.indptr[:-1],
                       aptre=matrix.indptr[1:],
                       asub=matrix.indices,
                       aval=matrix.data,
                       bkc=bkc,
                       blc=blc,
                       buc=buc,
                       bkx=bkx,
                       blx=blx,
                       bux=bux)


        # Solve the problem
        task.optimize()
        basic = mosek.soltype.bas
        sol = task.getsolution(basic)
        status = sol[1]
        xx = sol[6]
        yy = sol[7]


        # Get objective values, solutions x, dual values y
        if solve_dual:
            primal = task.getdualobj(basic) + c0
            dual = task.getprimalobj(basic) + c0
            x_values = dict(zip(variables, yy))
            y_values = [-x for x in xx]
        else:
            primal = task.getprimalobj(basic)
            dual = task.getdualobj(basic)
            x_values = dict(zip(variables, xx))
            y_values = [-y for y in yy]

        if status == mosek.solsta.optimal:
            status_str = "feasible"
        else:
            status_str = "infeasible"

        # Extract the certificate
        certificate = {x: 0 for x in known_vars}

        # Each monomial with known value is associated with a sum of duals
        for i, cons in enumerate(constraints):
            for x in set(cons).intersection(known_vars):
                certificate[x] += y_values[i] * cons[x]

        # Clean entries with coefficient zero
        for x in list(certificate):
            if np.isclose(certificate[x], 0):
                del certificate[x]

        return {
            "primal_value": primal,
            "dual_value": dual,
            "status": status_str,
            "dual_certificate": certificate,
            "x": x_values
        }


if __name__ == '__main__':
    simple_lp = {
        "objective": {'x': 1, 'y': 1, 'z': 1, 'w': -2},  # x + y + z - 2w
        "known_vars": {'1': 1},  # Define the variable that is the identity
        "inequalities": [{'x': -1, '1': 2},  # 2 - x >= 0
                         {'y': -1, '1': 5},  # 5 - y >= 0
                         {'z': -1, '1': 1 / 2},  # 1/2 - z >= 0
                         {'w': 1, '1': 1}],  # w >= -1
        "equalities": [{'x': 1 / 2, 'y': 2, '1': -3}],  # x/2 + 2y - 3 = 0
        "solve_dual": True
    }
    safe_sol = solveLP_MosekFUSION(**simple_lp)
    raw_sol = solveLP_Mosek(**simple_lp)
    print(safe_sol)
    print(raw_sol)
