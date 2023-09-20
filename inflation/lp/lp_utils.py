import sys
import mosek
import numpy as np

from typing import List, Dict, Union
from scipy.sparse import coo_matrix, issparse
from time import perf_counter
from gc import collect
from inflation.utils import partsextractor, expand_sparse_vec, vstack
try:
    import gurobipy as gp
except ModuleNotFoundError:
    pass


def drop_zero_rows(coo_mat: coo_matrix):
    nz_rows, new_row = np.unique(coo_mat.row, return_inverse=True)
    coo_mat.row[:] = new_row
    coo_mat._shape = (len(nz_rows), coo_mat.shape[1])

def canonical_order(coo_mat: coo_matrix):
    order = np.lexsort([coo_mat.col, coo_mat.row])
    coo_mat.row[:] = np.asarray(coo_mat.row)[order]
    coo_mat.col[:] = np.asarray(coo_mat.col)[order]
    coo_mat.data[:] = np.asarray(coo_mat.data)[order]

def solveLP(objective: Union[coo_matrix, Dict] = None,
            known_vars: Union[coo_matrix, Dict] = None,
            semiknown_vars: Dict = None,
            inequalities: Union[coo_matrix, List[Dict]] = None,
            equalities: Union[coo_matrix, List[Dict]] = None,
            lower_bounds: Union[coo_matrix, Dict] = None,
            upper_bounds: Union[coo_matrix, Dict] = None,
            solve_dual: bool = False,
            default_non_negative: bool = True,
            relax_known_vars: bool = False,
            relax_inequalities: bool = False,
            verbose: int = 0,
            solverparameters: Dict = None,
            variables: List = None,
            **kwargs
            ) -> Dict:
    """Wrapper function that converts all dictionaries to sparse matrices to
    pass to the solver.

    Parameters
    ----------
    objective : coo_matrix | Dict, optional
        Objective function
    known_vars : coo_matrix | Dict, optional
        Known values of the monomials
    semiknown_vars : Dict, optional
        Semiknown variables
    inequalities : coo_matrix | List[Dict], optional
        Inequality constraints
    equalities : coo_matrix | List[Dict], optional
        Equality constraints
    lower_bounds : coo_matrix | Dict, optional
        Lower bounds of variables
    upper_bounds : coo_matrix | Dict, optional
        Upper bounds of variables
    solve_dual : bool, optional
        Whether to solve the dual (``True``) or primal (``False``) formulation.
        By default, ``False``.
    default_non_negative : bool, optional
        Whether to set default primal variables as non-negative. By default,
        ``True``.
    relax_known_vars : bool, optional
        Do feasibility as optimization where each known value equality becomes
        two relaxed inequality constraints. E.g., P(A) = 0.7 becomes P(A) +
        lambda >= 0.7 and P(A) - lambda <= 0.7, where lambda is a slack
        variable. By default, ``False``.
    relax_inequalities : bool, optional
        Do feasibility as optimization where each inequality is relaxed by the
        non-negative slack variable lambda. By default, ``False``.
    verbose : int, optional
        Verbosity. Higher means more messages. By default, 0.
    solverparameters : dict, optional
        Parameters to pass to the MOSEK solver. For example, to control whether
        presolve is applied before optimization, set
        ``mosek.iparam.presolve_use`` to ``mosek.presolvemode.on`` or
        ``mosek.presolvemode.off``. Or, control which optimizer is used by
        setting an optimizer type to ``mosek.iparam.optimizer``. See `MOSEK's
        documentation
        <https://docs.mosek.com/latest/pythonapi/solver-parameters.html>`_ for
        more details.
    variables : list
        Monomials by name in same order as column indices of all other solver
        arguments

    Returns
    -------
    dict
        Primal objective value, dual objective value, problem status, success
        status, dual certificate (as dictionary and sparse matrix), x values,
        and response code.
    """
    # Save solver arguments
    solver_args = locals()

    # Check type for arguments related to the problem
    problem_args = ("objective",
                    "known_vars",
                    "semiknown_vars",
                    "inequalities",
                    "equalities",
                    "lower_bounds",
                    "upper_bounds")
    used_args = {k: v for k, v in solver_args.items()
                 if k in problem_args and v is not None}
    if all(issparse(arg) for arg in used_args.values()):
        assert variables is not None, "Variables must be declared when all " \
                                      "arguments are in sparse matrix form."
    elif all(isinstance(arg, (dict, list)) for arg in used_args.values()):
        if variables is None:
            # Infer variables
            variables = set()
            variables.update(objective)
            for ineq in inequalities:
                variables.update(ineq)
            for eq in equalities:
                variables.update(eq)
            for x, (c, x2) in semiknown_vars.items():
                variables.update([x, x2])
            variables.update(known_vars)
            variables = sorted(variables)
            solver_args["variables"] = variables
        solver_args.update(convert_dicts(**solver_args))
    else:
        assert variables is not None, "Variables must be declared when " \
                                      "arguments are of mixed form."
        solver_args.update(convert_dicts(**solver_args))
    solver_args.pop("semiknown_vars", None)
    return solveLP_sparse(**solver_args)

blank_coo_matrix = coo_matrix((0,0), dtype=np.int8)
def solveLP_sparse(objective: coo_matrix = blank_coo_matrix,
                   known_vars: coo_matrix = blank_coo_matrix,
                   inequalities: coo_matrix = blank_coo_matrix,
                   equalities: coo_matrix = blank_coo_matrix,
                   lower_bounds: coo_matrix = blank_coo_matrix,
                   upper_bounds: coo_matrix = blank_coo_matrix,
                   solve_dual: bool = False,
                   default_non_negative: bool = True,
                   relax_known_vars: bool = False,
                   relax_inequalities: bool = False,
                   verbose: int = 0,
                   solverparameters: Dict = None,
                   variables: List = None,
                   **kwargs
                   ) -> Dict:
    """Internal function to solve an LP with the Mosek Optimizer API using
    sparse matrices. Columns of each matrix correspond to a fixed order of
    variables in the LP.

    Parameters
    ----------
    objective : coo_matrix, optional
        Objective function with coefficients as matrix entries.
    known_vars : coo_matrix, optional
        Known values of the monomials with values as matrix entries.
    inequalities : coo_matrix, optional
        Inequality constraints in matrix form.
    equalities : coo_matrix, optional
        Equality constraints in matrix form.
    lower_bounds : coo_matrix, optional
        Lower bounds of variables with bounds as matrix entries.
    upper_bounds : coo_matrix, optional
        Upper bounds of variables with bounds as matrix entries.
    solve_dual : bool, optional
        Whether to solve the dual (``True``) or primal (``False``) formulation.
        By default, ``False``.
    default_non_negative : bool, optional
        Whether to set default primal variables as non-negative. By default,
        ``True``.
    relax_known_vars : bool, optional
        Do feasibility as optimization where each known value equality becomes
        two relaxed inequality constraints. E.g., P(A) = 0.7 becomes P(A) +
        lambda >= 0.7 and P(A) - lambda <= 0.7, where lambda is a slack
        variable. By default, ``False``.
    relax_inequalities : bool, optional
        Do feasibility as optimization where each inequality is relaxed by the
        non-negative slack variable lambda. By default, ``False``.
    verbose : int, optional
        Verbosity. Higher means more messages. By default, 0.
    solverparameters : dict, optional
        Parameters to pass to the MOSEK solver. For example, to control whether
        presolve is applied before optimization, set
        ``mosek.iparam.presolve_use`` to ``mosek.presolvemode.on`` or
        ``mosek.presolvemode.off``. Or, control which optimizer is used by
        setting an optimizer type to ``mosek.iparam.optimizer``. See `MOSEK's
        documentation
        <https://docs.mosek.com/latest/pythonapi/solver-parameters.html>`_ for
        more details.
    variables : list
        Monomials by name in same order as column indices of all other solver
        arguments

    Returns
    -------
    dict
        Primal objective value, dual objective value, problem status, success
        status, dual certificate (as dictionary and sparse matrix), x values,
        and response code.
    """

    drop_zero_rows(inequalities)
    drop_zero_rows(equalities)
    canonical_order(known_vars)

    if verbose > 1:
        t0 = perf_counter()
        t_total = perf_counter()
        print("Starting pre-processing for the LP solver...")

    # Since the value of infinity is ignored, define it for symbolic purposes
    inf = 0.0

    if relax_known_vars or relax_inequalities:
        default_non_negative = False

    with mosek.Env() as env:
        with mosek.Task(env) as task:
            # Set parameters for the solver depending on value type
            if solverparameters:
                for param, val in solverparameters.items():
                    if isinstance(val, int):
                        task.putintparam(param, val)
                    elif isinstance(val, float):
                        task.putdouparam(param, val)
                    elif isinstance(val, str):
                        task.putstrparam(param, val)
            if solve_dual:
                task.putintparam(mosek.iparam.sim_solve_form,
                                 mosek.solveform.dual)
            else:
                task.putintparam(mosek.iparam.sim_solve_form,
                                 mosek.solveform.primal)
            if verbose > 0:
                # Attach a log stream printer to the task
                task.set_Stream(mosek.streamtype.log, streamprinter)
                task.putintparam(mosek.iparam.log_include_summary,
                                 mosek.onoffkey.on)
                task.putintparam(mosek.iparam.log_storage, 1)
            if verbose < 2:
                task.putintparam(mosek.iparam.log_sim, 0)
                task.putintparam(mosek.iparam.log_intpnt, 0)

            # Initialize constraint matrix
            constraints = vstack((inequalities, equalities))
            # nof_primal_inequalities = 0
            # if inequalities.nnz:
            #     nof_primal_inequalities = inequalities.shape[0]
            # nof_primal_equalities = 0
            # if equalities.nnz:
            #     nof_primal_equalities = equalities.shape[0]

            nof_primal_inequalities = inequalities.shape[0]
            nof_primal_equalities = equalities.shape[0]

            (nof_primal_constraints, nof_primal_variables) = constraints.shape
            nof_known_vars = known_vars.nnz

            # Initialize b vector (RHS of constraints)
            b = [0] * nof_primal_constraints


            if relax_inequalities:
                # Add slack variable lambda to each inequality
                cons_row = np.hstack(
                    (constraints.row, np.arange(nof_primal_inequalities)))
                cons_col = np.hstack(
                    (constraints.col, np.repeat(nof_primal_variables,
                                                nof_primal_inequalities)))
                cons_data = np.hstack(
                    (constraints.data, np.repeat(1, nof_primal_inequalities)))
                constraints = coo_matrix((cons_data, (cons_row, cons_col)),
                                         shape=(nof_primal_constraints,
                                                nof_primal_variables + 1))

            if relax_known_vars:
                # Each known value is replaced by two inequalities with slacks
                # ELIE VERSION
                # kv_row = np.repeat(
                #     np.arange(nof_known_vars * 2),
                #     2)
                # kv_col = np.empty((nof_known_vars * 4,), dtype=int)
                # kv_col[0:(2*nof_known_vars):2] = np.arange(nof_known_vars)
                # kv_col[(2 * nof_known_vars):(4 * nof_known_vars):2] = kv_col[0:(2*nof_known_vars):2]
                # kv_col[1::2] = np.broadcast_to(nof_primal_variables, 2*nof_known_vars)
                # kv_data = np.hstack((
                #     np.broadcast_to(1, 2*nof_known_vars),
                #     np.tile([1, -1], nof_known_vars)
                # ))
                # ERICA VERSION
                kv_row = np.tile(
                    np.arange(nof_known_vars * 2),
                    2)
                kv_col = np.hstack((
                    np.tile(known_vars.col, 2),
                    np.broadcast_to(nof_primal_variables, nof_known_vars * 2)
                ))
                kv_data = np.hstack((
                    np.broadcast_to(1, nof_known_vars * 3),
                    np.broadcast_to(-1, nof_known_vars)
                ))

                kv_matrix = coo_matrix((kv_data, (kv_row, kv_col)),
                                       shape=(nof_known_vars * 2,
                                              nof_primal_variables + 1))
                canonical_order(kv_matrix)
                constraints.resize(*(nof_primal_constraints,
                                     nof_primal_variables + 1))
                b = np.hstack((b, np.tile(known_vars.data, 2)))
            else:
                # Add known values as equalities to the constraint matrix
                kv_matrix = expand_sparse_vec(known_vars)
                b = np.hstack((b, known_vars.data))
                nof_primal_equalities += nof_known_vars

            constraints = vstack((constraints, kv_matrix))
            (nof_primal_constraints, nof_primal_variables) = constraints.shape

            nof_lb = lower_bounds.nnz
            nof_ub = upper_bounds.nnz

            if verbose > 0:
                print(f"Size of constraint matrix: {constraints.shape}")

            if solve_dual:
                if verbose > 1:
                    print("Proceeding with dual initialization...")

                nof_primal_nontriv_bounds = nof_lb + nof_ub
                nof_dual_constraints = nof_primal_variables
                nof_dual_variables = nof_primal_constraints + \
                    nof_primal_nontriv_bounds

                # Add variable bounds as inequality constraints to matrix
                if nof_primal_nontriv_bounds > 0:
                    lb_mat = expand_sparse_vec(lower_bounds,
                                               conversion_style="eq")
                    ub_mat = expand_sparse_vec(upper_bounds,
                                               conversion_style="eq")
                    ub_mat.data[:] = -ub_mat.data # just a list of ones
                    # ub_data = -ub_mat.data
                    # ub_mat = coo_matrix((ub_data, (ub_mat.row, ub_mat.col)),
                    #                     shape=(nof_ub, nof_primal_variables))
                    matrix = vstack((constraints, lb_mat, ub_mat),
                                    format='csr')
                    b_extra = np.concatenate(
                        (lower_bounds.data, -np.asarray(upper_bounds.data)))
                    objective_vector = np.concatenate((b, b_extra))
                else:
                    matrix = constraints.tocsr(copy=False)
                    objective_vector = b

                if verbose > 1:
                    print("Sparse matrix reformat complete...")

                # Set bound keys and values for constraints (primal objective)
                if relax_known_vars or relax_inequalities:
                    blc = buc = np.zeros(nof_primal_variables)
                    blc[-1] = buc[-1] = -1
                else:
                    blc = buc = objective.toarray().ravel()

                # Set constraint bounds corresponding to primal variable bounds
                if default_non_negative:
                    bkc = [mosek.boundkey.lo] * nof_dual_constraints
                else:
                    bkc = [mosek.boundkey.fx] * nof_dual_constraints
                    # if relax_known_vars or relax_inequalities:
                    #     bkc[-1] = mosek.boundkey.lo

                # Set bound keys and values for variables
                # Non-positivity for y corresponding to inequalities
                bkx = [mosek.boundkey.up] * nof_primal_inequalities + \
                      [mosek.boundkey.fr] * nof_primal_equalities
                bux = [0.0] * nof_dual_variables
                blx = [0.0] * nof_dual_variables
                if relax_known_vars:
                    bkx.extend([mosek.boundkey.up] * nof_known_vars +
                               [mosek.boundkey.lo] * nof_known_vars)
                bkx.extend([mosek.boundkey.up] * nof_primal_nontriv_bounds)

                # Set the objective sense
                task.putobjsense(mosek.objsense.minimize)

            else:
                if verbose > 1:
                    print("Proceeding with primal initialization...")

                matrix = constraints.tocsc(copy=False)

                if verbose > 1:
                    print("Sparse matrix reformat complete...")

                if relax_known_vars or relax_inequalities:
                    # Minimize lambda
                    objective_vector = np.zeros(nof_primal_variables)
                    objective_vector[-1] = -1
                else:
                    objective_vector = objective.toarray().ravel()

                # Set bound keys and values for constraints
                # Ax >= b where b is 0
                bkc = [mosek.boundkey.lo] * nof_primal_inequalities + \
                      [mosek.boundkey.fx] * nof_primal_equalities
                if relax_known_vars:
                    bkc.extend([mosek.boundkey.lo] * nof_known_vars +
                               [mosek.boundkey.up] * nof_known_vars)
                blc = buc = b

                ub_col = upper_bounds.col
                ub_data = upper_bounds.toarray().ravel()
                lb_col = lower_bounds.col
                lb_data = lower_bounds.toarray().ravel()

                # Set bound keys and bound values for variables
                if default_non_negative:
                    blx = np.zeros(nof_primal_variables)
                    bkx = [mosek.boundkey.lo] * nof_primal_variables
                else:
                    blx = [-inf] * nof_primal_variables
                    bkx = [mosek.boundkey.fr] * nof_primal_variables
                bux = [+inf] * nof_primal_variables
                for col in range(nof_primal_variables):
                    if col in lb_col and col in ub_col:
                        bkx[col] = mosek.boundkey.ra
                        blx[col] = lb_data[col]
                        bux[col] = ub_data[col]
                    elif col in lb_col:
                        bkx[col] = mosek.boundkey.lo
                        blx[col] = lb_data[col]
                    elif col in ub_col:
                        bkx[col] = mosek.boundkey.up
                        bux[col] = ub_data[col]
                if relax_known_vars or relax_inequalities:
                    bkx[-1] = mosek.boundkey.fr

                # Set the objective sense
                task.putobjsense(mosek.objsense.maximize)

            # Add all the problem data to the task
            if solve_dual:
                numcon = nof_dual_constraints
                numvar = nof_dual_variables
            else:
                numcon = nof_primal_constraints
                numvar = nof_primal_variables
            if verbose > 1:
                print("Starting task.inputdata in Mosek...")
            task.inputdata(# maxnumcon=
                           numcon,
                           # maxnumvar=
                           numvar,
                           # c=
                           objective_vector,
                           # cfix=
                           0,
                           # aptrb=
                           matrix.indptr[:-1],
                           # aptre=
                           matrix.indptr[1:],
                           # asub=
                           matrix.indices,
                           # aval=
                           matrix.data,
                           bkc,
                           blc,
                           buc,
                           bkx,
                           blx,
                           bux)
            collect()
            if verbose > 1:
                print("Pre-processing took",
                      format(perf_counter() - t0, ".4f"), "seconds.\n")
                t0 = perf_counter()

            task.writedata("debug.ptf")

            # Solve the problem
            if verbose > 0:
                print("\nSolving the problem...\n")
            trmcode = task.optimize()
            if verbose > 1:
                print("Solving took", format(perf_counter() - t0, ".4f"),
                      "seconds.")
            basic = mosek.soltype.bas
            (problemsta,
             solutionsta,
             skc,
             skx,
             skn,
             xc,
             xx,
             yy,
             slc,
             suc,
             slx,
             sux,
             snx) = task.getsolution(basic)

            # Get objective values, solutions x, dual values y
            if solve_dual:
                primal = task.getdualobj(basic)
                dual = task.getprimalobj(basic)
                x_values = dict(zip(variables, yy))
                y_values = xx
            else:
                primal = task.getprimalobj(basic)
                dual = task.getdualobj(basic)
                x_values = dict(zip(variables, xx))
                y_values = yy

            if solutionsta == mosek.solsta.optimal:
                success = True
            else:
                success = False
            status_str = solutionsta.__repr__()
            term_tuple = mosek.Env.getcodedesc(trmcode)
            if solutionsta == mosek.solsta.unknown and verbose > 0:
                print("The solution status is unknown.")
                print(f"   Termination code: {term_tuple}")

            # Extract the certificate as a sparse matrix: y.b - c.x <= 0
            if relax_known_vars:
                y_values = y_values[nof_primal_constraints - nof_known_vars*2:]
            else:
                y_values = y_values[nof_primal_constraints - nof_known_vars:]
            cert_row = [0] * nof_primal_variables
            cert_col = [*range(nof_primal_variables)]
            cert_data = [0] * nof_primal_variables
            obj_data = objective.toarray().ravel()
            for col in np.setdiff1d(objective.col, known_vars.col):
                cert_data[col] -= obj_data[col]
            for i, col in enumerate(known_vars.col):
                cert_data[col] += y_values[i]
                if relax_known_vars:
                    cert_data[col] += y_values[i + nof_known_vars]
            sparse_certificate = coo_matrix((cert_data, (cert_row, cert_col)),
                                            shape=(1, nof_primal_variables))

            # Certificate as a dictionary
            certificate = dict(zip(variables,
                                   sparse_certificate.toarray().ravel()))

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
                "sparse_certificate": sparse_certificate,
                "x": x_values,
                "term_code": term_tuple
            }


def solve_Gurobi(objective: coo_matrix = coo_matrix([]),
                 known_vars: coo_matrix = coo_matrix([]),
                 inequalities: coo_matrix = coo_matrix([]),
                 equalities: coo_matrix = coo_matrix([]),
                 lower_bounds: coo_matrix = coo_matrix([]),
                 upper_bounds: coo_matrix = coo_matrix([]),
                 factorization_conditions: dict = None,
                 default_non_negative: bool = True,
                 relax_known_vars: bool = False,
                 relax_inequalities: bool = False,
                 verbose: int = 0,
                 **kwargs
                 ) -> Dict:
    """Internal function to solve an LP with the Gurobi Optimizer API using
    sparse matrices. Columns of each matrix correspond to a fixed order of
    variables in the LP.

    Parameters
    ----------
    objective : coo_matrix, optional
        Objective function with coefficients as matrix entries.
    known_vars : coo_matrix, optional
        Known values of the monomials with values as matrix entries.
    inequalities : coo_matrix, optional
        Inequality constraints in matrix form.
    equalities : coo_matrix, optional
        Equality constraints in matrix form.
    lower_bounds : coo_matrix, optional
        Lower bounds of variables with bounds as matrix entries.
    upper_bounds : coo_matrix, optional
        Upper bounds of variables with bounds as matrix entries.
    factorization_conditions : dict, optional
        Allows one to specify that certain variables are equal to the product of other variables
    default_non_negative : bool, optional
        Whether to set default primal variables as non-negative. By default,
        ``True``.
    relax_known_vars : bool, optional
        Do feasibility as optimization where each known value equality becomes
        two relaxed inequality constraints. E.g., P(A) = 0.7 becomes P(A) +
        lambda >= 0.7 and P(A) - lambda <= 0.7, where lambda is a slack
        variable. By default, ``False``.
    relax_inequalities : bool, optional
        Do feasibility as optimization where each inequality is relaxed by the
        non-negative slack variable lambda. By default, ``False``.
    verbose : int, optional
        Verbosity. Higher means more messages. By default, 0.

    Returns
    -------
    dict
        Primal objective value, problem status, success status, x values
    """
    if verbose > 1:
        t0 = perf_counter()
        t_total = perf_counter()
        print("Starting pre-processing for the LP solver...")

    env = gp.Env(empty=True)
    if verbose == 0:
        env.setParam('OutputFlag', 0)  # Supress output from solver
    env.start()

    # Create new model
    m = gp.Model(env=env)

    (nof_primal_inequalities, nof_primal_variables) = inequalities.shape
    nof_primal_equalities = equalities.shape[0]
    kv_matrix = expand_sparse_vec(known_vars)

    # Create variables and set variable bounds
    x = m.addMVar(shape=nof_primal_variables)
    m.update()
    if not default_non_negative:
        x.lb = -gp.GRB.INFINITY
    if lower_bounds.nnz > 0:
        new_lb = x.lb
        new_lb[lower_bounds.col] = lower_bounds.data
        x.lb = new_lb
    if upper_bounds.nnz > 0:
        new_ub = x.ub
        new_ub[upper_bounds.col] = upper_bounds.data
        x.ub = new_ub

    # Set objective
    objective_vector = objective.toarray().ravel()
    m.setObjective(objective_vector @ x, gp.GRB.MAXIMIZE)

    # Add inequality constraint matrix
    rhs_ineq = np.zeros(nof_primal_inequalities)
    if inequalities.size > 0:
        m.addConstr(inequalities @ x >= rhs_ineq, name="ineq")

    # Add equality constraint matrix
    rhs_eq = np.zeros(nof_primal_equalities)
    if equalities.size > 0:
        m.addConstr(equalities @ x == rhs_eq, name="eq")

    # Add known value constraint matrix
    rhs_kv = known_vars.data
    m.addConstr(kv_matrix @ x == rhs_kv, name="kv")

    # Add quadratic constraints
    if factorization_conditions:
        m.setParam("NonConvex", 2)
        m.addConstrs((x[mon] == x[factors[0]] * x[factors[1]]
                     for mon, factors in factorization_conditions.items()),
                     name="fac")

    collect()
    if verbose > 1:
        print("Pre-processing took",
              format(perf_counter() - t0, ".4f"), "seconds.\n")
        t0 = perf_counter()

    # Solve the problem
    if verbose > 0:
        print("\nSolving the problem...\n")
    m.optimize()
    if verbose > 1:
        print("Solving took", format(perf_counter() - t0, ".4f"),
              "seconds.")

    sc = gp.StatusConstClass
    solution_statuses = {sc.__dict__[k]: k for k in sc.__dict__.keys()
                         if 'A' <= k[0] <= 'Z'}
    status_str = solution_statuses[m.Status]
    if status_str == "OPTIMAL":
        success = True
        primal = m.ObjVal
        x_values = x.X
    else:
        success = False
        primal = None
        x_values = None

    if verbose > 1:
        print("\nTotal execution time:",
              format(perf_counter() - t_total, ".4f"), "seconds.")

    return {
        "primal_value": primal,
        "status": status_str,
        "success": success,
        "x": x_values,
    }


def solveLP_Mosek(objective: Dict = None,
                  known_vars: Dict = None,
                  semiknown_vars: Dict = None,
                  inequalities: List[Dict] = None,
                  equalities: List[Dict] = None,
                  lower_bounds: Dict = None,
                  upper_bounds: Dict = None,
                  solve_dual: bool = False,
                  default_non_negative: bool = True,
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
    known_vars : dict, optional
        Monomials (keys) and known values of the monomials
    semiknown_vars : dict, optional
        Encodes proportionality constraints between monomials
    inequalities : list of dict, optional
        Inequality constraints with monomials (keys) and coefficients (values)
    equalities : list of dict, optional
        Equality constraints with monomials (keys) and coefficients (values)
    lower_bounds : dict, optional
        Lower bounds of variables
    upper_bounds : dict, optional
        Upper bounds of variables
    solve_dual : bool, optional
        Whether to solve the dual (True) or primal (False) formulation
    default_non_negative : bool, optional
        Whether to set default primal variables as non-negative (True) or not
        (False)
    feas_as_optim : bool, optional
        NOT IMPLEMENTED
    verbose : int, optional
        verbosity. Higher means more messages. Default 0.
    solverparameters : dict, optional
        Parameters to pass to the MOSEK solver. For example, to control whether
        presolve is applied before optimization, set
        ``mosek.iparam.presolve_use`` to ``mosek.presolvemode.on`` or
        ``mosek.presolvemode.off``. Or, control which optimizer is used by
        setting an optimizer type to ``mosek.iparam.optimizer``. See `MOSEK's
        documentation
        <https://docs.mosek.com/latest/pythonapi/solver-parameters.html>`_ for
        more details.

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
    known_vars_as_set = set(known_vars)
    variables.update(known_vars_as_set)
    variables = sorted(variables)

    # Create dictionary var_index - monomial : index
    var_index = {x: i for i, x in enumerate(variables)}

    with mosek.Env() as env:
        with mosek.Task(env) as task:
            # Set parameters for the solver depending on value type
            if solverparameters:
                for param, val in solverparameters.items():
                    if isinstance(val, int):
                        task.putintparam(param, val)
                    elif isinstance(val, float):
                        task.putdouparam(param, val)
                    elif isinstance(val, str):
                        task.putstrparam(param, val)
            if solve_dual:
                task.putintparam(mosek.iparam.sim_solve_form,
                                 mosek.solveform.dual)
            else:
                task.putintparam(mosek.iparam.sim_solve_form,
                                 mosek.solveform.primal)
            if verbose > 0:
                # Attach a log stream printer to the task
                task.set_Stream(mosek.streamtype.log, streamprinter)
                task.putintparam(mosek.iparam.log_include_summary,
                                 mosek.onoffkey.on)
                task.putintparam(mosek.iparam.log_storage, 1)
            if verbose < 2:
                task.putintparam(mosek.iparam.log_sim, 0)
                task.putintparam(mosek.iparam.log_intpnt, 0)

            nof_primal_variables = len(variables)
            nof_primal_inequalities = len(inequalities)
            nof_primal_equalities = len(internal_equalities) + len(known_vars)
            nof_primal_constraints = nof_primal_inequalities + \
                nof_primal_equalities

            # Create sparse matrix A of constraints
            constraints = inequalities + internal_equalities

            Arow, Acol, Adata, brow, bcol, bdata = [], [], [], [], [], []
            for i, constraint in enumerate(constraints):
                var_part = {var_index[x]: v
                            for x, v in constraint.items()}
                Arow.extend([i]*len(var_part))
                for j, v in var_part.items():
                    Acol.append(j)
                    Adata.append(v)
                const_part = 0
                brow.append(i)
                bcol.append(0)
                bdata.append(const_part)
            for i, (x, b) in enumerate(known_vars.items()):
                Arow.append(len(constraints) + i)
                Acol.append(var_index[x])
                Adata.append(1)
                brow.append(len(constraints) + i)
                bcol.append(0)
                bdata.append(b)
            A = coo_matrix((Adata, (Arow, Acol)), shape=(nof_primal_constraints,
                                                         nof_primal_variables))
            b = coo_matrix((bdata, (brow, bcol)),
                           shape=(nof_primal_constraints,
                                  1)).toarray().ravel().tolist()

            if verbose > 0:
                print(f"Size of matrix A: {A.shape}")

            # Objective function coefficients
            c = np.zeros(nof_primal_variables)
            for x in set(objective).difference(known_vars_as_set):
                c[var_index[x]] = objective[x]

            # Compute c0, the constant (fixed) term in the objective function
            c0 = sum(objective[x] * known_vars[x]
                     for x in set(objective).intersection(known_vars_as_set))

            if solve_dual:
                if verbose>1:
                    print("Proceeding with dual initialization...")
                # matrix = A.tocsc(copy=False)

                nof_primal_lower_bounds = len(lower_bounds)
                nof_primal_nontriv_bounds = nof_primal_lower_bounds + \
                    len(upper_bounds)
                nof_dual_constraints = nof_primal_variables
                nof_dual_variables = nof_primal_equalities + \
                    nof_primal_inequalities + nof_primal_nontriv_bounds

                if nof_primal_nontriv_bounds > 0:
                    Arow = np.arange(nof_primal_nontriv_bounds)
                    Acol, Adata, b_extra = [], [], []
                    for mon, bnd in lower_bounds.items():
                        Acol.append(var_index[mon])
                        Adata.append(1)
                        b_extra.append(bnd)
                    for mon, bnd in upper_bounds.items():
                        Acol.append(var_index[mon])
                        Adata.append(-1)
                        b_extra.append(-bnd)
                    A_extra = coo_matrix((Adata, (Arow, Acol)),
                                         shape=(nof_primal_nontriv_bounds,
                                                nof_primal_variables))

                    matrix = vstack((A, A_extra), format='csr')
                    objective_vector = b + b_extra
                else:
                    matrix = A.tocsr(copy=False)
                    objective_vector = b
                if verbose > 1:
                    print("Sparse matrix reformat complete...")
                # Set bound keys and values for constraints (primal objective)
                blc = c
                buc = c

                # Set constraint bounds corresponding to primal variable bounds
                if default_non_negative:
                    bkc = [mosek.boundkey.lo] * nof_dual_constraints
                else:
                    bkc = [mosek.boundkey.fx] * nof_dual_constraints

                # Set bound keys and values for variables
                # Non-positivity for y corresponding to inequalities
                bkx = [mosek.boundkey.up] * nof_primal_inequalities + \
                      [mosek.boundkey.fr] * nof_primal_equalities + \
                      [mosek.boundkey.up] * nof_primal_nontriv_bounds
                blx = [-inf] * nof_dual_variables
                bux = [0.0] * nof_primal_inequalities + \
                      [+inf] * nof_primal_equalities + \
                      [0.0] * nof_primal_nontriv_bounds

                # Set the objective sense
                task.putobjsense(mosek.objsense.minimize)

            else:
                if verbose > 1:
                    print("Proceeding with primal initialization...")
                matrix = A.tocsc(copy=False)
                if verbose > 1:
                    print("Sparse matrix reformat complete...")
                objective_vector = c

                # Set bound keys and bound values (lower and upper) for constraints
                # Ax + b >= 0 -> Ax >= -b
                bkc = [mosek.boundkey.lo] * nof_primal_inequalities + \
                      [mosek.boundkey.fx] * nof_primal_equalities
                blc = buc = b

                # Set correct bounds if x >= 0
                if default_non_negative:
                    for x in variables:
                        if x in lower_bounds:
                            lower_bounds[x] = max(0, lower_bounds[x])
                        else:
                            lower_bounds[x] = 0

                # Set bound keys and bound values (lower and upper) for variables
                bkx = [mosek.boundkey.fr] * nof_primal_variables
                blx = [-inf] * nof_primal_variables
                bux = [+inf] * nof_primal_variables
                for x in set(lower_bounds).intersection(upper_bounds):
                    i = var_index[x]
                    bkx[i] = mosek.boundkey.ra
                    blx[i] = lower_bounds[x]
                    bux[i] = upper_bounds[x]
                for x in set(lower_bounds).difference(upper_bounds):
                    i = var_index[x]
                    bkx[i] = mosek.boundkey.lo
                    blx[i] = lower_bounds[x]
                for x in set(upper_bounds).difference(lower_bounds):
                    i = var_index[x]
                    bkx[i] = mosek.boundkey.up
                    bux[i] = upper_bounds[x]

                # Set the objective sense
                task.putobjsense(mosek.objsense.maximize)

            # Add all the problem data to the task
            if solve_dual:
                numcon = nof_dual_constraints
                numvar = nof_dual_variables
            else:
                numcon = nof_primal_constraints
                numvar = nof_primal_variables
            if verbose > 1:
                print("Starting task.inputdata in Mosek...")
            task.inputdata(# maxnumcon=
                           numcon,
                           # maxnumvar=
                           numvar,
                           # c=
                           objective_vector,
                           # cfix=
                           c0,
                           # aptrb=
                           matrix.indptr[:-1],
                           # aptre=
                           matrix.indptr[1:],
                           # asub=
                           matrix.indices,
                           # aval=
                           matrix.data,
                           bkc,
                           blc,
                           buc,
                           bkx,
                           blx,
                           bux)
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
            (problemsta,
             solutionsta,
             skc,
             skx,
             skn,
             xc,
             xx,
             yy,
             slc,
             suc,
             slx,
             sux,
             snx) = task.getsolution(basic)

            # Get objective values, solutions x, dual values y
            if solve_dual:
                primal = task.getdualobj(basic)
                dual = task.getprimalobj(basic)
                x_values = dict(zip(variables, yy))
                y_values = xx
            else:
                primal = task.getprimalobj(basic)
                dual = task.getdualobj(basic)
                x_values = dict(zip(variables, xx))
                y_values = yy

            if solutionsta == mosek.solsta.optimal:
                success = True
            else:
                success = False
            status_str = solutionsta.__repr__()
            term_tuple = mosek.Env.getcodedesc(trmcode)
            if solutionsta == mosek.solsta.unknown and verbose > 0:
                print("The solution status is unknown.")
                print(f"   Termination code: {term_tuple}")

            # Extract the certificate: y.b - c.x <= 0 for all primal feasible
            certificate = {x: 0 for x in variables}

            for x in set(objective).difference(known_vars):
                certificate[x] -= objective[x]
            for i, x in enumerate(known_vars):
                certificate[x] += y_values[len(constraints) + i]

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


###########################################################################
# ROUTINES RELATED TO SPARSE MATRIX CONVERSION                            #
###########################################################################


def to_sparse(argument: Union[Dict, List[Dict]],
              variables: List) -> coo_matrix:
    """Convert a solver argument to a sparse matrix to pass to the solver."""
    if type(argument) == dict:
        var_to_idx = {x: i for i, x in enumerate(variables)}
        data = list(argument.values())
        keys = list(argument.keys())
        col = partsextractor(var_to_idx, keys)
        row = np.zeros(len(col), dtype=int)
        return coo_matrix((data, (row, col)), shape=(1, len(variables)))
    else:
        # Argument is a list of constraints
        row = []
        for i, cons in enumerate(argument):
            row.extend([i] * len(cons))
        cols = [to_sparse(cons, variables).col for cons in argument]
        col = [c for vec_col in cols for c in vec_col]
        data = [to_sparse(cons, variables).data for cons in argument]
        data = [d for vec_data in data for d in vec_data]
        return coo_matrix((data, (row, col)),
                          shape=(len(argument), len(variables)))


def convert_dicts(objective: Union[coo_matrix, Dict] = None,
                  known_vars: Union[coo_matrix, Dict] = None,
                  semiknown_vars: Dict = None,
                  inequalities: Union[coo_matrix, List[Dict]] = None,
                  equalities: Union[coo_matrix, List[Dict]] = None,
                  lower_bounds: Union[coo_matrix, Dict] = None,
                  upper_bounds: Union[coo_matrix, Dict] = None,
                  solve_dual: bool = False,
                  default_non_negative: bool = True,
                  relax_known_vars: bool = False,
                  relax_inequalities: bool = False,
                  verbose: int = 0,
                  solverparameters: Dict = None,
                  variables: List = None) -> Dict:
    """Convert any dictionaries to sparse matrices to send to the solver.
    Semiknown variables are absorbed into the equality constraints. Only
    converted arguments are returned."""

    assert variables is not None, "Variables must be passed."

    # Arguments converted from dictionaries to sparse matrices
    sparse_args = {k: to_sparse(arg, variables) for k, arg in locals().items()
                   if isinstance(arg, (dict, list)) and k != "semiknown_vars"
                   and k != "solverparameters" and k != "variables"}

    # Add semiknown variables to equality constraints
    if semiknown_vars:
        nof_semiknown = len(semiknown_vars)
        nof_variables = len(variables)
        var_to_idx = {x: i for i, x in enumerate(variables)}
        row = np.repeat(np.arange(nof_semiknown), 2)
        col = [(var_to_idx[x], var_to_idx[x2])
               for x, (c, x2) in semiknown_vars.items()]
        col = list(sum(col, ()))
        data = [(1, -c) for x, (c, x2) in semiknown_vars.items()]
        data = list(sum(data, ()))
        semiknown_mat = coo_matrix((data, (row, col)),
                                   shape=(nof_semiknown, nof_variables))
        if "equalities" in sparse_args:
            sparse_args["equalities"] = vstack(
                (sparse_args["equalities"], semiknown_mat))
        else:
            sparse_args["equalities"] = semiknown_mat
    return sparse_args


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
    raw_sol = solveLP_Mosek(**simple_lp, solve_dual=False)
    raw_sol_d = solveLP_Mosek(**simple_lp, solve_dual=True)

    obj = coo_matrix(([-2, 1, 1, 1], ([0, 0, 0, 0], [1, 2, 3, 4])),
                     shape=(1, 5))
    kv = coo_matrix(([1], ([0], [0])), shape=(1, 5))
    ineq = coo_matrix(([2, -1, 5, -1, 0.5, -1, 1, 1],
                       ([0, 0, 1, 1, 2, 2, 3, 3], [0, 2, 0, 3, 0, 4, 0, 1])),
                      shape=(4, 5))
    eq = coo_matrix(([-3, 0.5, 2], ([0, 0, 0], [0, 2, 3])), shape=(1, 5))
    simple_lp_mat = {
        "objective": obj,
        "known_vars": kv,
        "inequalities": ineq,
        "equalities": eq,
        "variables": ['1', 'w', 'x', 'y', 'z']
    }
    mat_sol = solveLP_sparse(solve_dual=False, **simple_lp_mat)
    mat_sol_d = solveLP_sparse(solve_dual=True, **simple_lp_mat)
