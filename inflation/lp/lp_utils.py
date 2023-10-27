"""
This file contains functions to interact with LP solvers.

@authors: Erica Han, Elie Wolfe
"""

import sys
import mosek
import numpy as np

from typing import List, Dict, Union
from scipy.sparse import coo_matrix, issparse, hstack
from time import perf_counter
from gc import collect
from inflation.utils import partsextractor, expand_sparse_vec, vstack


def drop_zero_rows(coo_mat: coo_matrix):
    """Drops zero rows from a sparse matrix in place.
    
    Parameters
    ----------
    coo_mat : coo_matrix
        Sparse matrix to drop zero rows from.
    """
    nz_rows, new_row = np.unique(coo_mat.row, return_inverse=True)
    coo_mat.row[:] = new_row
    coo_mat._shape = (len(nz_rows), coo_mat.shape[1])


def canonical_order(coo_mat: coo_matrix):
    """Puts a sparse matrix in canonical order in place.
    
    Parameters
    ----------
    coo_mat : coo_matrix
        Sparse matrix to put in canonical order.
    """
    order = np.lexsort([coo_mat.col, coo_mat.row])
    coo_mat.row[:] = np.asarray(coo_mat.row)[order]
    coo_mat.col[:] = np.asarray(coo_mat.col)[order]
    coo_mat.data[:] = np.asarray(coo_mat.data)[order]


def solveLP(objective: Union[coo_matrix, Dict] = None,
            known_vars: Union[coo_matrix, Dict] = None,
            semiknown_vars: Dict = None,
            inequalities: Union[coo_matrix, List[Dict]] = None,
            equalities: Union[coo_matrix, List[Dict]] = None,
            variables: List = None,
            **kwargs
            ) -> Dict:
    """Wrapper function that converts all dictionaries to sparse matrices to
    pass to the solver.

    Parameters
    ----------
    objective : Union[coo_matrix, Dict], optional
        Objective function
    known_vars : Union[coo_matrix, Dict], optional
        Known values of the monomials
    semiknown_vars : Dict, optional
        Semiknown variables
    inequalities : Union[coo_matrix, List[Dict]], optional
        Inequality constraints
    equalities : Union[coo_matrix, List[Dict]], optional
        Equality constraints
    variables : List
        Monomials by name in same order as column indices of all other solver
        arguments

    Returns
    -------
    dict
        Primal objective value, dual objective value, problem status, success
        status, dual certificate (as dictionary and sparse matrix), x values,
        and response code.
    """
    # Save solver arguments, unpacking kwargs
    solver_args = locals()
    del solver_args['kwargs']
    solver_args.update(kwargs)

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
            if objective:
                variables.update(objective)
            if inequalities:
                for ineq in inequalities:
                    variables.update(ineq)
            if equalities:
                for eq in equalities:
                    variables.update(eq)
            if semiknown_vars:
                for x, (c, x2) in semiknown_vars.items():
                    variables.update([x, x2])
            if known_vars:
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


blank_coo_matrix = coo_matrix((0, 0), dtype=np.int8)
def solveLP_sparse(objective: coo_matrix = blank_coo_matrix,
                   known_vars: coo_matrix = blank_coo_matrix,
                   inequalities: coo_matrix = blank_coo_matrix,
                   equalities: coo_matrix = blank_coo_matrix,
                   lower_bounds: coo_matrix = blank_coo_matrix,
                   upper_bounds: coo_matrix = blank_coo_matrix,
                   solve_dual: bool = False,
                   default_non_negative: bool = True,
                   relaxation: bool = False,
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
    default_non_negative : bool, optional
        Whether to set default primal variables as non-negative. By default,
        ``True``.
    relaxation : bool, optional
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

    if relaxation:
        default_non_negative = False

        # Add slack variable lambda to each inequality
        lambda_col = np.repeat(1, inequalities.shape[0])[:, None]
        inequalities = hstack((inequalities, lambda_col))
        variables = np.append(variables, "\u03bb")  # lambda as Unicode

    # Initialize constraint matrix
    constraints = vstack((inequalities, equalities))
    if verbose > 0:
        print(f"Size of constraint matrix: {constraints.shape}")

    nof_primal_inequalities = inequalities.shape[0]
    nof_primal_equalities = equalities.shape[0]
    (nof_primal_constraints, nof_primal_variables) = constraints.shape
    nof_known_vars = known_vars.nnz
    nof_lb = lower_bounds.nnz
    nof_ub = upper_bounds.nnz

    # Initialize b vector (RHS of constraints)
    b = np.zeros(nof_primal_constraints)

    if verbose > 1:
        print("Proceeding with primal initialization...")

    matrix = constraints.tocsc(copy=False)

    if verbose > 1:
        print("Sparse matrix reformat complete...")

    if relaxation:
        # Minimize lambda
        objective_vector = np.zeros(nof_primal_variables)
        objective_vector[-1] = -1
    else:
        objective_vector = objective.toarray().ravel()

    # Set bound keys and values for constraints: Ax >= b where b = 0
    bkc = np.concatenate((
        np.repeat(mosek.boundkey.lo, nof_primal_inequalities),
        np.repeat(mosek.boundkey.fx, nof_primal_equalities)))
    blc = buc = b

    # Create bkx, lx, ux (representing variable bounds) from known values,
    #   lower bounds, and upper bounds
    if default_non_negative:
        lb_values = np.zeros(nof_primal_variables)
    else:
        lb_values = np.empty(nof_primal_variables, dtype=object)
    ub_values = np.empty(nof_primal_variables, dtype=object)
    kv_values = np.empty(nof_primal_variables, dtype=object)
    bkx = np.empty(nof_primal_variables, dtype=mosek.boundkey)
    blx = np.zeros(nof_primal_variables)
    bux = np.zeros(nof_primal_variables)

    lb_values[lower_bounds.col] = lower_bounds.data
    ub_values[upper_bounds.col] = upper_bounds.data
    kv_values[known_vars.col] = known_vars.data

    for i, (lb, ub, kv) in enumerate(zip(lb_values, ub_values, kv_values)):
        if kv is not None:
            bkx[i] = mosek.boundkey.fx
            blx[i] = bux[i] = kv
        elif lb is not None and ub is not None:
            bkx[i] = mosek.boundkey.ra
            blx[i] = lb
            bux[i] = ub
        elif lb is not None:
            bkx[i] = mosek.boundkey.lo
            blx[i] = lb
        elif ub is not None:
            bkx[i] = mosek.boundkey.up
            bux[i] = ub
        else:
            bkx[i] = mosek.boundkey.fr

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
                task.putintparam(mosek.iparam.intpnt_solve_form,
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

            # Set the objective sense
            task.putobjsense(mosek.objsense.maximize)

            # Add all the problem data to the task
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

            if verbose > 1:
                print("Writing problem to debug_lp.ptf...")
                task.writedata("debug_lp.ptf")

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

            # Get objective values and solutions x
            primal = task.getprimalobj(basic)
            dual = task.getdualobj(basic)
            x_values = dict(zip(variables, xx))
            y_values = np.asarray(yy)

            if solutionsta == mosek.solsta.optimal:
                success = True
            else:
                success = False
            status_str = solutionsta.__repr__()
            term_tuple = mosek.Env.getcodedesc(trmcode)
            if solutionsta == mosek.solsta.unknown and verbose > 0:
                print("The solution status is unknown.")
                print(f"   Termination code: {term_tuple}")

            # Extract the certificate as a string: c.x >= (slx-sux).x
            variables_arr = np.asarray(variables)
            problem_specific_vars = set(variables_arr[known_vars.col]) \
                .union(variables_arr[lower_bounds.col],
                       variables_arr[upper_bounds.col],
                       variables_arr[objective.col])
            # Create x_sym which contains variable as a string for
            #   problem-specific variables, else as numeric value
            x_sym = np.empty(nof_primal_variables, dtype=object)
            for i in range(nof_primal_variables):
                if variables[i] in problem_specific_vars:
                    x_sym[i] = variables[i]
                else:
                    x_sym[i] = xx[i]

            cTx_str, sTx_str = "", ""
            for i in range(nof_primal_variables):
                if not np.isclose(objective_vector[i], 0):
                    cTx_str += f"{objective_vector[i]}*{x_sym[i]} + "
                if not np.isclose(slx[i]-sux[i], 0):
                    sTx_str += f"{slx[i]-sux[i]}*{x_sym[i]} + "
            cTx_str = cTx_str[:-3]
            if cTx_str == "":
                cTx_str = "0"
            sTx_str = sTx_str[:-3]
            if solve_dual:
                prob_cert = cTx_str + " <= " + sTx_str
            else:
                prob_cert = cTx_str + " >= " + sTx_str

            # Extract the certificate as a sparse matrix: (slx-sux).b - c.x <= 0
            cert_row = np.zeros(nof_primal_variables)
            cert_col = np.arange(nof_primal_variables)
            cert_data = np.zeros(nof_primal_variables)
            obj_data = objective_vector
            objective_unknown_cols = np.setdiff1d(objective.col, known_vars.col)
            cert_data[objective_unknown_cols] = -obj_data[objective_unknown_cols]
            cert_data[known_vars.col] = (np.asarray(slx)-np.asarray(sux))[known_vars.col]
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
                "term_code": term_tuple,
                "problem_specific_certificate": prob_cert
            }


def streamprinter(text: str) -> None:
    """A stream printer to get output from Mosek.
    
    Parameters
    ----------
    text : str
        Text to print.
    """
    sys.stdout.write(text)
    sys.stdout.flush()


###########################################################################
# ROUTINES RELATED TO SPARSE MATRIX CONVERSION                            #
###########################################################################


def to_sparse(argument: Union[Dict, List[Dict]],
              variables: List) -> coo_matrix:
    """Convert a solver argument to a sparse matrix to pass to the solver.
    
    Parameters
    ----------
    argument : Union[Dict, List[Dict]]
        Solver argument to convert to sparse matrix.
    variables : List
        Monomials by name in same order as column indices of all other solver
        arguments
    
    Returns
    -------
    coo_matrix
        Sparse matrix representation of the solver argument.
    """
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


def convert_dicts(semiknown_vars: Dict = None,
                  variables: List = None,
                  **kwargs) -> Dict:
    """Convert any dictionaries to sparse matrices to send to the solver.
    Semiknown variables are absorbed into the equality constraints.
    
    Parameters
    ----------
    semiknown_vars : Dict, optional
        Semiknown variables
    variables : List
        Monomials by name in same order as column indices of all other solver
        arguments
    
    Returns
    -------
    Dict
        Converted arguments (Unchanged arguments are not returned)
    """

    assert variables is not None, "Variables must be passed."

    args = locals()
    del args['kwargs']
    args.update(kwargs)

    # Arguments converted from dictionaries to sparse matrices
    sparse_args = {k: to_sparse(arg, variables) for k, arg in args.items()
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
