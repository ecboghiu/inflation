import sys
import mosek
import numpy as np

from typing import List, Dict
from scipy.sparse import vstack, coo_matrix
from time import perf_counter
from gc import collect


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

        nof_primal_variables = len(variables)
        nof_primal_inequalities = len(inequalities)
        nof_primal_equalities = len(internal_equalities)
        nof_primal_constraints = nof_primal_inequalities + \
            nof_primal_equalities

        # Create sparse matrix A of constraints
        constraints = inequalities + internal_equalities

        Arow, Acol, Adata, brow, bcol, bdata = [], [], [], [], [], []
        for i, constraint in enumerate(constraints):
            constraint_vars = set(constraint)
            for x in constraint_vars.difference(known_vars):
                Arow.append(i)
                Acol.append(var_index[x])
                Adata.append(constraint[x])
            for x in constraint_vars.intersection(known_vars):
                brow.append(i)
                bcol.append(0)
                bdata.append(-constraint[x] * known_vars[x])
        A = coo_matrix((Adata, (Arow, Acol)), shape=(nof_primal_constraints,
                                                     nof_primal_variables))
        b = coo_matrix((bdata, (brow, bcol)),
                       shape=(nof_primal_constraints,
                              1)).toarray().ravel().tolist()

        if verbose > 0:
            print(f"Size of matrix A: {A.shape}")

        # Objective function coefficients
        c = np.zeros(nof_primal_variables)
        for x in set(objective).difference(known_vars):
            c[var_index[x]] = objective[x]

        # Compute c0, the constant (fixed) term in the objective function
        c0 = 0
        for x in set(objective).intersection(known_vars):
            c0 += objective[x] * known_vars[x]

        if solve_dual:
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
                matrix = A.asformat('csr', copy=False)
                objective_vector = b

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
            matrix = A.asformat('csc', copy=False)
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
            for x, i in var_index.items():
                if x in set(lower_bounds).intersection(upper_bounds):
                    bkx[i] = mosek.boundkey.ra
                    blx[i] = lower_bounds[x]
                    bux[i] = upper_bounds[x]
                elif x in lower_bounds:
                    bkx[i] = mosek.boundkey.lo
                    blx[i] = lower_bounds[x]
                elif x in upper_bounds:
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
            y_values = [-x for x in xx]
        else:
            primal = task.getprimalobj(basic)
            dual = task.getdualobj(basic)
            x_values = dict(zip(variables, xx))
            y_values = [-y for y in yy]

        if solutionsta == mosek.solsta.optimal:
            success = True
        else:
            success = False
        status_str = solutionsta.__repr__()
        term_tuple = mosek.Env.getcodedesc(trmcode)
        if solutionsta == mosek.solsta.unknown and verbose > 0:
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
    raw_sol = solveLP_Mosek(**simple_lp, solve_dual=False)
    raw_sol_d = solveLP_Mosek(**simple_lp, solve_dual=True)
