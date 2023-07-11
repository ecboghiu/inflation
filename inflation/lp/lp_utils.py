import sys
import mosek
import numpy as np

from typing import List, Dict, Union
from scipy.sparse import vstack, coo_matrix
from time import perf_counter
from gc import collect
from inflation.utils import partsextractor, sparse_vec_to_sparse_mat


def solveLP(objective: Union[coo_matrix, Dict],
            known_vars: Union[coo_matrix, Dict],
            semiknown_vars: Dict,
            inequalities: Union[coo_matrix, List[Dict]],
            equalities: Union[coo_matrix, List[Dict]],
            lower_bounds: Union[coo_matrix, Dict],
            upper_bounds: Union[coo_matrix, Dict],
            solve_dual: bool = False,
            all_non_negative: bool = True,
            feas_as_optim: bool = False,
            verbose: int = 0,
            solverparameters: Dict = None,
            variables: List = None
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
    all_non_negative : bool, optional
        Whether to set all primal variables as non-negative. By default,
        ``True``.
    feas_as_optim : bool, optional
        NOT IMPLEMENTED
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
    used_args = {k: v for k, v in solver_args
                 if k in problem_args and v is not None}
    if all(isinstance(arg, coo_matrix) for arg in used_args.values()):
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
        solver_args.update(convert_dicts(**used_args, variables=variables))
    else:
        assert variables is not None, "Variables must be declared when " \
                                      "arguments are of mixed form."
        solver_args.update(convert_dicts(**used_args, variables=variables))
    solver_args.pop("semiknown_vars", None)
    return solveLP_sparse(**solver_args)


def solveLP_sparse(objective: coo_matrix = coo_matrix([]),
                   known_vars: coo_matrix = coo_matrix([]),
                   inequalities: coo_matrix = coo_matrix([]),
                   equalities: coo_matrix = coo_matrix([]),
                   lower_bounds: coo_matrix = coo_matrix([]),
                   upper_bounds: coo_matrix = coo_matrix([]),
                   solve_dual: bool = False,
                   all_non_negative: bool = True,
                   feas_as_optim: bool = False,
                   verbose: int = 0,
                   solverparameters: Dict = None,
                   variables: List = None
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
    all_non_negative : bool, optional
        Whether to set all primal variables as non-negative. By default,
        ``True``.
    feas_as_optim : bool, optional
        NOT IMPLEMENTED
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

    if verbose > 1:
        t0 = perf_counter()
        t_total = perf_counter()
        print("Starting pre-processing for the LP solver...")

    # Since the value of infinity is ignored, define it for symbolic purposes
    inf = 0.0

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
            (nof_primal_constraints, nof_primal_variables) = constraints.shape

            # Initialize b vector (RHS of constraints)
            b = [0] * nof_primal_constraints

            # Add known values as equality constraints to the constraint matrix
            kv_matrix = sparse_vec_to_sparse_mat(known_vars)
            constraints = vstack((constraints, kv_matrix))
            b.extend(known_vars.data)

            (nof_primal_constraints, nof_primal_variables) = constraints.shape
            nof_known_vars = known_vars.nnz
            nof_primal_inequalities = inequalities.shape[0]
            nof_primal_equalities = equalities.shape[0] + nof_known_vars
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
                    lb_mat = sparse_vec_to_sparse_mat(lower_bounds)
                    ub_mat = sparse_vec_to_sparse_mat(upper_bounds)
                    ub_data = [-1] * nof_ub
                    ub_mat = coo_matrix((ub_data, (ub_mat.row, ub_mat.col)),
                                        shape=(nof_ub, nof_primal_variables))
                    matrix = vstack((constraints, lb_mat, ub_mat),
                                    format='csr')
                    b_extra = np.concatenate(
                        (lower_bounds.data, [-ub for ub in upper_bounds.data]))
                    objective_vector = np.concatenate((b, b_extra))
                else:
                    matrix = constraints.tocsr(copy=False)
                    objective_vector = b

                if verbose > 1:
                    print("Sparse matrix reformat complete...")

                # Set bound keys and values for constraints (primal objective)
                blc = objective.toarray().ravel()
                buc = objective.toarray().ravel()

                # Set constraint bounds corresponding to primal variable bounds
                if all_non_negative:
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

                matrix = constraints.tocsc(copy=False)

                if verbose > 1:
                    print("Sparse matrix reformat complete...")

                objective_vector = objective.toarray().ravel()

                # Set bound keys and values for constraints
                # Ax + b >= 0 -> Ax >= -b
                bkc = [mosek.boundkey.lo] * nof_primal_inequalities + \
                      [mosek.boundkey.fx] * nof_primal_equalities
                blc = buc = b

                # Set correct bounds if x >= 0
                if all_non_negative:
                    zeros = np.zeros(nof_primal_variables)
                    lb_col = [*range(nof_primal_variables)]
                    # If there are no lower bounds, resize creates array of
                    # zeros. If there are lower bounds, negative bounds are
                    # corrected to zero.
                    lb_data = np.maximum(
                        np.resize(lower_bounds.toarray(),
                                  (1, nof_primal_variables)),
                        zeros).ravel()
                else:
                    lb_col = lower_bounds.col
                    lb_data = lower_bounds.toarray().ravel()
                ub_col = upper_bounds.col
                ub_data = upper_bounds.toarray().ravel()

                # Set bound keys and bound values for variables
                bkx = [mosek.boundkey.fr] * nof_primal_variables
                blx = [-inf] * nof_primal_variables
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
                        bux[col] = mosek.boundkey.up
                        bux[col] = ub_data[col]

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

            # Extract the certificate as a sparse matrix: c⋅x - y⋅b <= 0
            y_values = y_values[nof_primal_constraints - nof_known_vars:]
            cert_row = [0] * nof_primal_variables
            cert_col = [*range(nof_primal_variables)]
            cert_data = [0] * nof_primal_variables
            obj_data = objective.toarray().ravel()
            for col in np.setdiff1d(objective.col, known_vars.col):
                cert_data[col] -= obj_data[col]
            for i, col in enumerate(known_vars.col):
                cert_data[col] += y_values[i]
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


def solveLP_Mosek(objective: Dict = None,
                  known_vars: Dict = None,
                  semiknown_vars: Dict = None,
                  inequalities: List[Dict] = None,
                  equalities: List[Dict] = None,
                  lower_bounds: Dict = None,
                  upper_bounds: Dict = None,
                  solve_dual: bool = False,
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
    all_non_negative : bool, optional
        Whether to set all primal variables as non-negative (True) or not
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
                if all_non_negative:
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
                if all_non_negative:
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

            # Extract the certificate: y.b - c⋅x >= 0 for all primal feasible
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


def dict_to_sparse_vec(str_dict: Dict, variables: List) -> coo_matrix:
    """Convert a dictionary of monomial names and values to a one-dimensional
    sparse matrix.
    """
    var_to_idx = {x: i for i, x in enumerate(variables)}
    data = list(str_dict.values())
    keys = list(str_dict.keys())
    col = partsextractor(var_to_idx, keys)
    row = np.zeros(len(col), dtype=int)
    return coo_matrix((data, (row, col)), shape=(1, len(variables)))


def constraint_vec_to_mat(constraints: List[Dict],
                          variables: List) -> coo_matrix:
    """Convert a list of dictionaries representing constraints to a sparse
    matrix."""
    row = []
    for i, cons in enumerate(constraints):
        row.extend([i] * len(cons))
    cols = [dict_to_sparse_vec(cons, variables).col for cons in constraints]
    col = [c for vec_col in cols for c in vec_col]
    data = [dict_to_sparse_vec(cons, variables).data for cons in constraints]
    data = [d for vec_data in data for d in vec_data]
    return coo_matrix((data, (row, col)),
                      shape=(len(constraints), len(variables)))


def convert_dicts(objective: Union[coo_matrix, Dict] = None,
                  known_vars: Union[coo_matrix, Dict] = None,
                  semiknown_vars: Union[coo_matrix, Dict] = None,
                  inequalities: Union[coo_matrix, List[Dict]] = None,
                  equalities: Union[coo_matrix, List[Dict]] = None,
                  lower_bounds: Union[coo_matrix, Dict] = None,
                  upper_bounds: Union[coo_matrix, Dict] = None,
                  variables: List = None) -> Dict:
    """Convert any dictionaries to sparse matrices to send to the solver."""
    # Dictionary of arguments to convert
    sparse_args = {k: None for k, arg in locals().items()
                   if isinstance(arg, (dict, list))}
    if "objective" in sparse_args:
        sparse_args["objective"] = dict_to_sparse_vec(objective, variables)
    if "known_vars" in sparse_args:
        sparse_args["known_vars"] = dict_to_sparse_vec(known_vars, variables)
    if "inequalities" in sparse_args:
        sparse_args["inequalities"] = constraint_vec_to_mat(inequalities,
                                                            variables)
    if "equalities" in sparse_args:
        equalities_mat = constraint_vec_to_mat(equalities, variables)
        if "semiknown_vars" in sparse_args:
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
            equalities_mat = vstack((equalities_mat, semiknown_mat))
        sparse_args["equalities"] = equalities_mat
    if "lower_bounds" in sparse_args:
        sparse_args["lower_bounds"] = dict_to_sparse_vec(lower_bounds,
                                                         variables)
    if "upper_bounds" in sparse_args:
        ub = dict_to_sparse_vec(upper_bounds, variables)
        ub_data = [-c for c in ub.data]
        sparse_args["upper_bounds"] = coo_matrix((ub_data, (ub.row, ub.col)),
                                                 shape=(1, len(variables)))
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
