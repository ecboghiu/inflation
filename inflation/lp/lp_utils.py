"""
This file contains functions to interact with LP solvers.

@authors: Erica Han, Elie Wolfe
"""

import sys
import mosek
import numpy as np

from typing import List, Dict, Union
from scipy.sparse import coo_array, issparse
from time import perf_counter
from gc import collect
from ..utils import partsextractor, expand_sparse_vec, vstack


def drop_zero_rows(coo_mat: coo_array):
    """Drops zero rows from a sparse matrix in place.

    Parameters
    ----------
    coo_mat : coo_array
        Sparse matrix to drop zero rows from.
    """
    if len(coo_mat.shape) == 1:
        coo_mat = coo_mat.reshape((1, coo_mat.shape[0]))
    nz_rows, new_row = np.unique(coo_mat.row, return_inverse=True)
    coo_mat.row = new_row
    coo_mat = coo_mat.reshape((len(nz_rows), coo_mat.shape[1]))
    return coo_mat


def canonical_order(coo_mat: coo_array):
    """Puts a sparse matrix in canonical order in place.

    Parameters
    ----------
    coo_mat : coo_array
        Sparse matrix to put in canonical order.
    """
    if len(coo_mat.shape) == 1:
        coo_mat = coo_mat.reshape((1, coo_mat.shape[0]))
    order = np.lexsort([coo_mat.col, coo_mat.row])
    coo_mat.row = np.asarray(coo_mat.row)[order]
    coo_mat.col = np.asarray(coo_mat.col)[order]
    coo_mat.data = np.asarray(coo_mat.data)[order]
    return coo_mat

def solveLP(objective: Union[coo_array, Dict] = None,
            known_vars: Union[coo_array, Dict] = None,
            semiknown_vars: Dict = None,
            inequalities: Union[coo_array, List[Dict]] = None,
            equalities: Union[coo_array, List[Dict]] = None,
            variables: List = None,
            **kwargs
            ) -> Dict:
    """Wrapper function that converts all dictionaries to sparse matrices to
    pass to the solver.

    Parameters
    ----------
    objective : Union[coo_array, Dict], optional
        Objective function
    known_vars : Union[coo_array, Dict], optional
        Known values of the monomials
    semiknown_vars : Dict, optional
        Semiknown variables
    inequalities : Union[coo_array, List[Dict]], optional
        Inequality constraints
    equalities : Union[coo_array, List[Dict]], optional
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


blank_coo_array = coo_array((0, 0), dtype=np.int8)
def solveLP_sparse(objective: coo_array = blank_coo_array,
                   known_vars: coo_array = blank_coo_array,
                   inequalities: coo_array = blank_coo_array,
                   equalities: coo_array = blank_coo_array,
                   lower_bounds: coo_array = blank_coo_array,
                   upper_bounds: coo_array = blank_coo_array,
                   solve_dual: bool = False,
                   default_non_negative: bool = True,
                   relax_known_vars: bool = False,
                   relax_inequalities: bool = False,
                   verbose: int = 0,
                   solverparameters: Dict = None,
                   variables: List = None
                   ) -> Dict:
    """Internal function to solve an LP with the Mosek Optimizer API using
    sparse matrices. Columns of each matrix correspond to a fixed order of
    variables in the LP.

    Parameters
    ----------
    objective : coo_array, optional
        Objective function with coefficients as matrix entries.
    known_vars : coo_array, optional
        Known values of the monomials with values as matrix entries.
    inequalities : coo_array, optional
        Inequality constraints in matrix form.
    equalities : coo_array, optional
        Equality constraints in matrix form.
    lower_bounds : coo_array, optional
        Lower bounds of variables with bounds as matrix entries.
    upper_bounds : coo_array, optional
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

    inequalities=drop_zero_rows(inequalities)
    equalities=drop_zero_rows(equalities)
    known_vars=canonical_order(known_vars)
    upper_bounds=canonical_order(upper_bounds)
    lower_bounds=canonical_order(lower_bounds)
    objective=canonical_order(objective)

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
                constraints = coo_array((cons_data, (cons_row, cons_col)),
                                         shape=(nof_primal_constraints,
                                                nof_primal_variables + 1))

            if relax_known_vars:
                # Each known value is replaced by two inequalities with slacks
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

                kv_matrix = coo_array((kv_data, (kv_row, kv_col)),
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
            if objective.shape[-1] == 0:
                objective = coo_array((1, nof_primal_variables), dtype=np.int8)

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
                    bkc = np.broadcast_to(mosek.boundkey.lo, nof_dual_constraints)
                else:
                    bkc = np.broadcast_to(mosek.boundkey.fx, nof_dual_constraints)

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
                    # Maximize lambda
                    # (If maximum slack is still negative, then the unrelaxed
                    # LP would be infeasible, whereas positive slack solution
                    # implies LP solution strictly interior in the polytope.)
                    objective_vector = np.zeros(nof_primal_variables)
                    objective_vector[-1] = -1
                else:
                    objective_vector = objective.toarray().ravel()

                # Set bound keys and values for constraints
                # Ax >= b where b is 0
                bkc = np.hstack((np.broadcast_to(mosek.boundkey.lo, nof_primal_inequalities),
                                 np.broadcast_to(mosek.boundkey.fx,
                                                 nof_primal_equalities)
                                 ))
                if relax_known_vars:
                    bkc = np.hstack((bkc,
                                     np.repeat([mosek.boundkey.lo, mosek.boundkey.up],
                                                     nof_known_vars)
                                     ))
                blc = buc = b

                ub_col = upper_bounds.col
                ub_data = np.zeros(nof_primal_variables)
                ub_data[ub_col] = upper_bounds.data
                lb_col = lower_bounds.col
                lb_data = np.zeros(nof_primal_variables)
                lb_data[lb_col] = lower_bounds.data

                # Set bound keys and bound values for variables
                blx = np.zeros(nof_primal_variables)
                bux = np.zeros(nof_primal_variables)
                ub_col = np.asarray(upper_bounds.col)
                lb_col = np.asarray(lower_bounds.col)
                lb_data = lower_bounds.data
                if default_non_negative:
                    bkx = np.repeat(mosek.boundkey.lo, nof_primal_variables)
                    bkx[ub_col] = mosek.boundkey.ra
                else:
                    bkx = np.repeat(mosek.boundkey.fr, nof_primal_variables)
                    bkx[np.setdiff1d(lb_col, ub_col)] = mosek.boundkey.lo
                    bkx[np.setdiff1d(ub_col, lb_col)] = mosek.boundkey.up
                    bkx[np.intersect1d(ub_col, lb_col)] = mosek.boundkey.ra
                blx[lb_col] = lb_data
                bux[ub_col] = upper_bounds.data


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

            # Get objective values, solutions x, dual values y
            if solve_dual:
                primal = task.getdualobj(basic)
                dual = task.getprimalobj(basic)
                x_values = dict(zip(variables, yy))
                y_values = np.asarray(xx)
            else:
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

            # Extract the certificate as a sparse matrix: y.b - c.x <= 0
            if relax_known_vars:
                y_values = y_values[nof_primal_constraints - nof_known_vars*2:]
            else:
                y_values = y_values[nof_primal_constraints - nof_known_vars:]
            cert_row = [0] * nof_primal_variables
            cert_col = [*range(nof_primal_variables)]
            cert_data = np.zeros((nof_primal_variables,))
            obj_data = objective.toarray().ravel()
            objective_unknown_cols = np.setdiff1d(objective.col, known_vars.col)
            cert_data[objective_unknown_cols] = -obj_data[objective_unknown_cols]
            cert_data[known_vars.col] = y_values[:nof_known_vars] # Assumes known values coded as equalities
            if relax_known_vars:
                cert_data[known_vars.col] += y_values[nof_known_vars:(2*nof_known_vars)]
            sparse_certificate = coo_array((cert_data, (cert_row, cert_col)),
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
              variables: List) -> coo_array:
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
    coo_array
        Sparse matrix representation of the solver argument.
    """
    if type(argument) == dict:
        var_to_idx = {x: i for i, x in enumerate(variables)}
        data = list(argument.values())
        keys = list(argument.keys())
        col = partsextractor(var_to_idx, keys)
        row = np.zeros(len(col), dtype=int)
        return coo_array((data, (row, col)), shape=(1, len(variables)))
    else:
        # Argument is a list of constraints
        row = []
        for i, cons in enumerate(argument):
            row.extend([i] * len(cons))
        cols = [to_sparse(cons, variables).col for cons in argument]
        col = [c for vec_col in cols for c in vec_col]
        data = [to_sparse(cons, variables).data for cons in argument]
        data = [d for vec_data in data for d in vec_data]
        return coo_array((data, (row, col)),
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
        semiknown_mat = coo_array((data, (row, col)),
                                   shape=(nof_semiknown, nof_variables))
        if "equalities" in sparse_args:
            sparse_args["equalities"] = vstack(
                (sparse_args["equalities"], semiknown_mat))
        else:
            sparse_args["equalities"] = semiknown_mat
    return sparse_args
