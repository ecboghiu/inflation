import sys
import mosek
import numpy as np

from typing import List, Dict
from mosek.fusion import Matrix, Model, ObjectiveSense, Expr, Domain, \
    OptimizeError, SolutionError, AccSolutionStatus, ProblemStatus
from scipy.sparse import dok_matrix, vstack
from time import perf_counter
from gc import collect

def solveLP_MosekFUSION(objective: Dict = None,
                        known_vars: Dict = None,
                        semiknown_vars: Dict = None,
                        inequalities: List[Dict] = None,
                        equalities: List[Dict] = None,
                        lower_bounds: Dict = None,
                        upper_bounds: Dict = None,
                        solve_dual: bool = True,
                        all_non_negative: bool = True,
                        feas_as_optim: bool = False,
                        verbose: int = 0,
                        solverparameters: Dict = None
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
        Equality constraints with monomials (keys) and coefficients (values)
    lower_bounds : dict
        Lower bounds of variables
    upper_bounds : dict
        Upper bounds of variables
    solve_dual : bool
        Whether to solve the dual (True) or primal (False) formulation
    all_non_negative : bool
        Whether to set all primal variables as non-negative (True) or not
        (False)
    feas_as_optim: bool
        NOT IMPLEMENTED
    verbose: bool
        NOT IMPLEMENTED
    solverparameters: dict
        NOT IMPLEMENTED

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

    # Absorb lower and upper bounds of variables into inequalities
    if lower_bounds:
        inequalities.append(lower_bounds)
    if upper_bounds:
        inequalities.append(upper_bounds)

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

            if all_non_negative:
                c = M.constraint("c",
                                 Expr.sub(Expr.mul(transpose, y), v_mosek),
                                 Domain.greaterThan(0.0))
            else:
                c = M.constraint("c",
                                 Expr.sub(Expr.mul(transpose, y), v_mosek),
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
            if all_non_negative:
                x = M.variable("x", nof_variables, Domain.greaterThan(0.0))
            else:
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
        elif status in [ProblemStatus.DualInfeasible,
                        ProblemStatus.PrimalInfeasible]:
            status_str = "infeasible"
        elif status == ProblemStatus.Unknown:
            status_str = "unknown"
        else:
            status_str = "other"

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
                  lower_bounds: Dict = None,
                  upper_bounds: Dict = None,
                  solve_dual: bool = True,
                  all_non_negative: bool = True,
                  feas_as_optim: bool = False,
                  verbose: int = 0,
                  solverparameters: Dict = None
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
        Equality constraints with monomials (keys) and coefficients (values)
    lower_bounds : dict
        Lower bounds of variables
    upper_bounds : dict
        Upper bounds of variables
    solve_dual : bool
        Whether to solve the dual (True) or primal (False) formulation
    all_non_negative : bool
        Whether to set all primal variables as non-negative (True) or not
        (False)
    feas_as_optim: bool
        NOT IMPLEMENTED
    verbose: int
        verbosity. Higher means more messages. Default 0.
    solverparameters: dict
        NOT IMPLEMENTED

    Returns
    -------
    dict
        Primal objective value, dual objective value, problem status, dual
        certificate, x values
    """

    if verbose > 1:
        t0 = perf_counter()
        t_total = perf_counter()
        print("Starting pre-processing for the LP solver...")

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
    if lower_bounds is None:
        lower_bounds = {}
    if upper_bounds is None:
        upper_bounds = {}

    # Absorb lower and upper bounds of variables into inequalities
    # TODO: Use these cleverly in primal/dual formulation??
    inequalities.extend({mon: 1, '1': -bnd}
                        for mon, bnd in lower_bounds.items())
    inequalities.extend({mon: -1, '1': bnd}
                        for mon, bnd in upper_bounds.items())


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
        if verbose > 0:
            # Attach a log stream printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)
            task.putintparam(mosek.iparam.log_include_summary,
                             mosek.onoffkey.on)
            task.putintparam(mosek.iparam.log_storage, 1)
        if verbose < 2:
            task.putintparam(mosek.iparam.log_sim, 0)
            task.putintparam(mosek.iparam.log_intpnt, 0)


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
        if verbose > 0:
            print(f"Size of matrix A: {A.get_shape()}")

        # Objective function coefficients
        c = np.zeros(numvar)
        for x in set(objective).difference(known_vars):
            c[var_index[x]] = objective[x]

        # Compute c0, the constant (fixed) term in the objective function
        c0 = 0
        for x in set(objective).intersection(known_vars):
            c0 += objective[x] * known_vars[x]

        if solve_dual:
            numcon = len(variables)
            numvar = len(inequalities + internal_equalities)
            matrix = A.asformat('csr', copy=False)
            objective_vector = b

            # Set bound keys and values for constraints (primal objective)
            if all_non_negative:
                bkc = [mosek.boundkey.lo] * numcon
            else:
                # All equalities since primal variable x is free
                bkc = [mosek.boundkey.fx] * numcon
            blc = buc = c

            # Set bound keys and values for variables
            # Non-positivity for y corresponding to inequalities
            bkx = [mosek.boundkey.up] * nof_inequalities + \
                  [mosek.boundkey.fr] * nof_equalities
            blx = [-inf] * numvar
            bux = [0.0] * nof_inequalities + [+inf] * nof_equalities

            # Set the objective sense
            task.putobjsense(mosek.objsense.minimize)

        else:
            matrix = A.asformat('csc', copy=False)
            objective_vector = c

            # Set bound keys and bound values (lower and upper) for constraints
            # Ax + b >= 0 -> Ax >= -b
            bkc = [mosek.boundkey.lo] * nof_inequalities + \
                  [mosek.boundkey.fx] * nof_equalities
            blc = buc = b

            # Set bound keys and bound values (lower and upper) for variables
            if all_non_negative:
                bkx = [mosek.boundkey.lo] * numvar
                blx = [0.0] * numvar
            else:
                bkx = [mosek.boundkey.fr] * numvar
                blx = [-inf] * numvar
            bux = [+inf] * numvar

            # Set the objective sense
            task.putobjsense(mosek.objsense.maximize)

        # Add all the problem data to the task
        task.inputdata(maxnumcon=numcon,
                       maxnumvar=numvar,
                       c=objective_vector,
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

        collect()
        if verbose > 1:
            print("Pre-processing took", format(perf_counter() - t0, ".4f"),
                  "seconds.\n")
            t0 = perf_counter()

        # Solve the problem
        if verbose > 0:
            print("\nSolving the problem...\n")
        trmcode = task.optimize()
        if verbose > 1:
            print("Solving took", format(perf_counter() - t0, ".4f"),
                  "seconds.")
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
            success = True
        else:
            success = False
        status_str = status.__repr__()
        term_tuple = mosek.Env.getcodedesc(trmcode)
        if status == mosek.solsta.unknown and verbose > 0:
            print("The solution status is unknown.")
            print(f"   Termination code: {term_tuple}")

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

        if verbose > 1:
            print("\nTotal execution time:",
                  format(perf_counter() - t_total, ".4f"), "seconds.")

        return {
            "primal_value": primal,
            "dual_value": dual,
            "status": status_str,
            "success": success,
            "dual_certificate": certificate,
            "x": x_values,
            "term_code": term_tuple
        }


def streamprinter(text: str) -> None:
    """A stream printer to get output from Mosek."""
    sys.stdout.write(text)
    sys.stdout.flush()


if __name__ == '__main__':
    simple_lp = {
        "objective": {'x': 1, 'y': 1, 'z': 1, 'w': -2},  # x + y + z - 2w
        "known_vars": {'1': 1},  # Define the variable that is the identity
        "inequalities": [{'x': -1, '1': 2},  # 2 - x >= 0
                         {'y': -1, '1': 5},  # 5 - y >= 0
                         {'z': -1, '1': 1 / 2},  # 1/2 - z >= 0
                         {'w': 1, '1': 1}],  # w >= -1
        "equalities": [{'x': 1 / 2, 'y': 2, '1': -3}],  # x/2 + 2y - 3 = 0
    }
    safe_sol = solveLP_MosekFUSION(**simple_lp,
                                   lower_bounds={'x': 0, 'y': 0,
                                                 'z': 0, 'w': 0})
    raw_sol = solveLP_Mosek(**simple_lp, all_non_negative=True)
    print(safe_sol)
    print(raw_sol)
