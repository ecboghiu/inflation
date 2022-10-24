"""
This file contains the functions to send the problems to SDP solvers.
@authors: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens
"""
import numpy as np

from scipy.sparse import dok_matrix, eye, lil_matrix
from sys import stdout
from typing import List, Dict, Tuple


def solveSDP_MosekFUSION(mask_matrices: Dict = None,
                         objective: Dict = None,
                         known_vars: Dict = None,
                         semiknown_vars: Dict = None,
                         inequalities: List[Dict] = None,
                         var_equalities: List[Dict] = None,
                         solve_dual: bool = True,
                         feas_as_optim: bool = False,
                         verbose: int = 0,
                         solverparameters: Dict = {},
                         process_constraints: bool = True
                         ) -> Tuple[Dict, float, str]:
    r"""Internal function to solve the SDP with the `MOSEK Fusion API
    <https://docs.mosek.com/latest/pythonfusion/index.html>`_.

    Now follows an extended description of how the SDP is encoded. In general,
    it is prefered to solve using the dual formulation, which is the default.

    The primal is written as follows:

    .. math::

        \text{max}\quad & c_0 + c\cdot x\\
        \text{s.t.}\quad & F_0 + \sum F_i x_i \succeq 0

    :math:`F_0` is the constant entries of the moment matrix, and :math:`F_i`
    is the matrix whose entry :math:`(n,m)` stores the value of the coefficient
    of the moment :math:`x_i` at position :math:`(n,m)` in the moment matrix.

    The dual of the equation above is:

    .. math::

        \text{min}\quad & c_0+\text{Tr}(Z\cdot F_0)\\
        \text{s.t.}\quad & \text{Tr}(Z\cdot F_i) = - c_i \,\forall\, i,\\
        &Z \succeq 0.

    Typically, all the probability information is stored in :math:`F_0`, and
    the coefficients :math:`F_i` do not depend on the probabilities. However,
    if we use LPI constraints (see, e.g., `arXiv:2203.16543
    <http://www.arxiv.org/abs/2203.16543/>`_), then :math:`F_i` can depend on
    the probabilities. The form of the SDP does not change, in any case.

    If we have a constant objective function, then we have a feasibility
    problem. It can be rewritten into the following optimization problem:

    .. math::
        \text{max}\quad&\lambda\\
        \text{s.t.}\quad& F_0 + \sum F_i x_i - \lambda \cdot 1 \succeq 0,

    which achieves :math:`\lambda\geq 0` if the original problem is feasible
    and :math:`\lambda<0` otherwise. The dual of this problem is:

    .. math::
        \text{min}\quad & \text{Tr}(Z\cdot F_0) \\
        \text{s.t.}\quad & \text{Tr}(Z\cdot F_i) = 0 \,\forall\, i,\\
            & Z \succeq 0,\,\text{Tr} Z = 1.

    This still allows for the extraction of certificates. If we use a
    :math:`Z_{P_1}` obtained from running the problem above on the probability
    distribution :math:`P_1`, and we find that
    :math:`\text{Tr}[Z_{P_1}\cdot F_0(P_2)] < 0`, then clearly this is an upper
    bound of the optimal value of the problem, and thus we can certify that the
    optimisation will be negative when using :math:`P_2`.

    If we have upper and lower bounds on the variables, the problems change as
    follows:

    .. math::
        \text{max}\quad & c_0 + c\cdot x \\
        \text{s.t.}\quad & F_0 + \sum F_i x_i \succeq 0,\\
        & x_i - l_i \geq 0,\\
        & u_i - x_i \geq 0,

    with dual:

    .. math::
        \text{min}\quad & \text{Tr}(Z\cdot F_0 - L\cdot l + U\cdot u) \\
        \text{s.t.}\quad & \text{Tr}(Z \cdot F_i) = -c_i+U_i-L_i\,\forall\,i,\\
        & Z \succeq 0,\,L \geq 0,\,U \geq 0.

    The relaxed feasibility problems change accordingly.

    Parameters
    ----------
    mask_matrices : dict
        A dictionary with keys as monomials and values as scipy sparse arrays
        indicating the locations of the monomial in the moment matrix.
    objective : dict, optional
        Dictionary with keys as monomials and as values the monomial's
        coefficient in the objective function.
    known_vars : dict, optional
        Dictionary of values for monomials (keys).
    semiknown_vars : dict, optional
        Dictionary encoding proportionality constraints between
        different monomials.
    inequalities : list, optional
        List of inequalities encoded as dictionaries of coefficients.
    var_equalities : list, optional
        List of equalities encoded as dictionaries of coefficients.
    solve_dual : bool, optional
        Whether to solve the dual (True) or primal (False) formulation. By
        default ``True``.
    feas_as_optim : bool, optional
        Whether to treat feasibility problems, where the objective is,
        constant, as an optimisation problem. By default ``False``.
    verbose : int, optional
        How much information to display to the user. By default ``0``.
    solverparameters : dict, optional
        Dictionary of parameters to pass to the MOSEK solver, see `MOSEK's
        documentation
        <https://docs.mosek.com/latest/pythonfusion/solver-parameters.html>`_.
    process_constraints: bool, optional
        Whether to remove the simple equalities constraints contained
        in the `semiknown_vars` arguments by eliminating variables (True)
        or pass them to the solver as equality constraints (False). By
        default ``True``.

    Returns
    -------
    Tuple[Dict, float, str]
        The first element of the tuple is a dictionary containing the
        optimisation information such as the 1) primal objective value,
        2) the moment matrix, 3) the dual values, 4) the certificate and
        a 5) dictionary of values for the monomials, in the following keys in
        the same order: 1) ``sol``, 2) ``G``, 3) ``Z``, 4)
        ``dual_certificate``, 5) ``xi``. The second element is the objective
        value and the last is the problem status.
    """
    import mosek
    from mosek.fusion import Matrix, Model, ObjectiveSense, Expr, Domain, \
        OptimizeError, SolutionError, \
        AccSolutionStatus, ProblemStatus

    if verbose > 1:
        from time import perf_counter
        t0 = perf_counter()
        print("Starting pre-processing for the SDP solver.")

    if mask_matrices is None:
        mask_matrices = {}
    if objective is None:
        objective = {}
    if known_vars is None:
        known_vars = {}
    if semiknown_vars is None:
        semiknown_vars = {}
    if inequalities is None:
        inequalities = []
    if var_equalities is None:
        var_equalities = []

    Fi = mask_matrices
    if mask_matrices:
        Fi = {k: lil_matrix(v, dtype=float) for k, v in Fi.items()}
        mat_dim = Fi[next(iter(Fi))].shape[0] if Fi else 0

    variables = set()
    variables.update(Fi.keys())
    variables.update(objective.keys())
    for eq in var_equalities:
        variables.update(eq.keys())
    for ineq in inequalities:
        variables.update(ineq.keys())
    variables.difference_update(known_vars.keys())

    if feas_as_optim and mask_matrices:
        if verbose > 0:
            print("Feasibility as optimisation detected: overwriting objective"
                  + " to maximize the smallest eigenvalue of the matrix"
                  + " variable.")
        lam = "lambda"
        while lam in variables:
            lam += "_"
        variables.add(lam)
        Fi[lam] = -1 * eye(mat_dim).tolil()
        objective = {lam: 1}

    var2index = {x: i for i, x in enumerate(variables)}

    # Calculate c0, the constant part of the objective.
    c0 = 0. + float(sum([objective[x] * known_vars[x]
                         for x in set(objective).intersection(known_vars)]))

    # Calculate F0, the constant part of the matrix variable.
    if mask_matrices:
        F0 = lil_matrix((mat_dim, mat_dim), dtype=float) + \
            sum([known_vars[x] * Fi[x]
                for x in set(Fi).intersection(known_vars)])

    # Calculate the matrices A, C and vectors b, d such that
    # Ax + b >= 0, Cx + d == 0.
    A = dok_matrix((len(inequalities), len(variables)))
    b = dok_matrix((len(inequalities), 1))
    for i, inequality in enumerate(inequalities):
        ineq_vars = set(inequality)
        for x in ineq_vars.difference(known_vars):
            A[i, var2index[x]] = inequality[x]
        b[i, 0] = float(sum([inequality[x] * known_vars[x]
                             for x in ineq_vars.intersection(known_vars)]))

    C = dok_matrix((len(var_equalities), len(variables)))
    d = dok_matrix((len(var_equalities), 1))
    for i, equality in enumerate(var_equalities):
        eq_vars = set(equality)
        for x in eq_vars.difference(known_vars):
            C[i, var2index[x]] = equality[x]
        d[i, 0] = float(sum([equality[x] * known_vars[x]
                             for x in eq_vars.intersection(known_vars)]))

    if process_constraints:
        # For the semiknown constraint x_i = a_i * x_j, add to the Fi of x_j
        # the expression a_i*(Fi of x_i).
        for x, (c, x2) in semiknown_vars.items():
            Fi[x2] += c * Fi[x]
            variables.remove(x)
            del Fi[x]

            if x in objective:
                objective[x2] = objective.get(x2, 0) + c * objective[x]
            for equality in var_equalities:
                if x in equality:
                    equality[x2] = equality.get(x2, 0) + c * equality[x]
            for inequality in inequalities:
                if x in inequality:
                    inequality[x2] = inequality.get(x2, 0) + c * inequality[x]
    else:
        # Just add semiknown constraints as equality constraints.
        for x, (c, x2) in semiknown_vars.items():
            var_equalities.append({x: 1, x2: -c})

    if mask_matrices:
        # These indices seem to be more difficult to extract later.
        ij_F0_nonzero = [(int(i), int(j)) for (i, j) in zip(*F0.nonzero()) if j >= i]
        ij_Fi_nonzero = {x: [(int(i), int(j))
                             for (i, j) in zip(*Fi[x].nonzero()) if j >= i]
                         for x in Fi}

    if verbose > 1:
        print("Pre-processing took",
              format(perf_counter() - t0, ".4f"),
              "seconds.")
        t0 = perf_counter()
    if verbose > 0:
        print("Building the model...")

    with Model("SDP") as M:
        if solve_dual:
            # Define variables
            if mask_matrices:
                Z = M.variable("Z", Domain.inPSDCone(mat_dim))
            if inequalities:
                I = M.variable("I", len(inequalities), Domain.greaterThan(0))
                # It seems MOSEK Fusion API does not allow to pick index i
                # of an expression (A^T I)_i, so we do it manually row by row.
                A = A.tocsr()
                AtI = []  # \sum_j I_j A_ji as i-th entry of AtI
                for i in range(len(variables)):
                    slice_ = A[:, i]
                    sparse_slice = Matrix.sparse(*slice_.shape,
                                                 *slice_.nonzero(),
                                                 slice_[slice_.nonzero()].A[0])
                    AtI.append(Expr.dot(sparse_slice, I))
            if var_equalities:
                E = M.variable("E", len(var_equalities), Domain.unbounded())
                C = C.tocsr()
                CtI = []  # \sum_j E_j C_ji as i-th entry of CtI
                for i in range(len(variables)):
                    slice_ = C[:, i]
                    sparse_slice = Matrix.sparse(*slice_.shape,
                                                 *slice_.nonzero(),
                                                 slice_[slice_.nonzero()].A[0])
                    CtI.append(Expr.dot(sparse_slice, E))

            # Define and set objective function
            # c0 + Tr Z F0 + I·b + E·d
            obj_mosek = 0.0
            if not feas_as_optim:
                obj_mosek = float(c0)
            if mask_matrices:
                F0_mosek = Matrix.sparse(*F0.shape,
                                         *F0.nonzero(),
                                         F0[F0.nonzero()].A[0])
                obj_mosek = Expr.add(obj_mosek, Expr.dot(Z, F0_mosek))
                del F0_mosek
            if inequalities:
                b_mosek = Matrix.sparse(*b.shape,
                                        *b.nonzero(),
                                        b[b.nonzero()].A[0])
                obj_mosek = Expr.add(obj_mosek, Expr.dot(I, b_mosek))
                del b_mosek
            if var_equalities:
                d_mosek = Matrix.sparse(*d.shape,
                                        *d.nonzero(),
                                        d[d.nonzero()].A[0])
                obj_mosek = Expr.add(obj_mosek, Expr.dot(E, d_mosek))
                del d_mosek

            M.objective(ObjectiveSense.Minimize, obj_mosek)

            # Add constraints
            # ci + Tr Z Fi + \sum_j I_j A_ji + \sum_j E_j C_ji == 0
            ci_constraints = []
            for i, x in enumerate(variables):
                lhs = 0.0
                if objective and x in objective:
                    ci  = float(objective[x])
                    lhs += ci
                if mask_matrices and x in Fi:
                    F = Fi[x]
                    F = Matrix.sparse(*F.shape,
                                      *F.nonzero(),
                                      F[F.nonzero()].A[0])
                    lhs = Expr.add(lhs, Expr.dot(Z, F))
                    del F
                if inequalities:
                    lhs = Expr.add(lhs, AtI[i])
                    AtI[i] = None
                if var_equalities:
                    lhs = Expr.add(lhs, CtI[i])
                    CtI[i] = None
                ci_constraints.append(M.constraint(f"c{i}",
                                                   lhs,
                                                   Domain.equalsTo(0)))
        else:
            # Set up the problem in primal formulation

            # Define variables
            x_mosek = M.variable("x", len(variables), Domain.unbounded())

            if inequalities:
                b_mosek = Matrix.sparse(*b.shape,
                                        *b.nonzero(),
                                        b[b.nonzero()].A[0])
                A_mosek = Matrix.sparse(*A.shape,
                                        *A.nonzero(),
                                        A[A.nonzero()].A[0])
                ineq_constraint = M.constraint("Ineq",
                                               Expr.add(Expr.mul(A_mosek,
                                                                 x_mosek),
                                                        b_mosek),
                                               Domain.greaterThan(0))
                del b_mosek, A_mosek
            if var_equalities:
                d_mosek = Matrix.sparse(*d.shape,
                                        *d.nonzero(),
                                        d[d.nonzero()].A[0])
                C_mosek = Matrix.sparse(*C.shape,
                                        *C.nonzero(),
                                        C[C.nonzero()].A[0])
                eq_constraint = M.constraint("Eq",
                                             Expr.add(Expr.mul(C_mosek,
                                                               x_mosek),
                                                      d_mosek),
                                             Domain.equalsTo(0))
                del d_mosek, C_mosek

            if mask_matrices:
                G = M.variable("G", Domain.inPSDCone(mat_dim))

                F0_mosek = Matrix.sparse(*F0.shape,
                                         *F0.nonzero(),
                                         F0[F0.nonzero()].A[0])
                Fi_mosek = {x: Matrix.sparse(*F.shape,
                                             *F.nonzero(),
                                             F[F.nonzero()].A[0])
                            for x, F in Fi.items()}

                # Add matrix constraints
                constraints = np.empty((mat_dim, mat_dim), dtype=object)
                for i in range(mat_dim):
                    for j in range(i, mat_dim):
                        constraints[i, j] = G.index(i, j)
                for i, j in ij_F0_nonzero:
                    constraints[i, j] = Expr.sub(constraints[i, j],
                                                 F0_mosek.get(i, j))
                for i, xi in enumerate(variables.intersection(Fi)):
                    for i_, j_ in ij_Fi_nonzero[xi]:
                        constraints[i_, j_] = \
                            Expr.sub(constraints[i_, j_],
                                     Expr.mul(Fi_mosek[xi].get(i_, j_),
                                              x_mosek.index(i)))

                for i in range(mat_dim):
                    for j in range(i, mat_dim):
                        # G(i,j) - F0(i,j) - sum_i xi Fi(i,j) = 0
                        M.constraint(constraints[i, j], Domain.equalsTo(0))

                del F0_mosek, Fi_mosek

            mosek_obj = c0
            for x in set(objective).difference(known_vars):
                ci = float(objective[x])
                mosek_obj = Expr.add(mosek_obj,
                                     Expr.mul(ci, x_mosek.index(var2index[x])))

            M.objective(ObjectiveSense.Maximize, mosek_obj)

        if verbose > 1:
            print("Model built in",
                  format(perf_counter() - t0, ".4f"),
                  "seconds.")
            M.writeTask("InflationSDPModel.ptf")
            print("Model saved to InflationSDPModel.ptf.")
            t0 = perf_counter()

        # Solve the problem and process the solution
        if verbose > 0:
            print("Solving the model...")
        xmat, ymat, primal, dual = None, None, None, None
        try:
            if verbose > 0:
                M.setLogHandler(stdout)
            if solverparameters:
                for param, val in solverparameters.items():
                    M.setSolverParam(param, val)
            M.acceptedSolutionStatus(AccSolutionStatus.Anything)
            M.solve()
            if solve_dual:
                x_values = {x: -ci_constraints[i].dual()[0]
                            for i, x in enumerate(variables)}
                if mask_matrices:
                    ymat = Z.level().reshape([mat_dim, mat_dim])
                    xmat = F0 + sum([x_values[x] * Fi[x]
                                     for x in set(Fi).difference(known_vars)])
            else:
                x_values = dict(zip(variables, x_mosek.level()))
                if mask_matrices:
                    ymat = G.dual().reshape([mat_dim, mat_dim])
                    xmat = G.level().reshape([mat_dim, mat_dim])

            status = M.getProblemStatus()
            if status == ProblemStatus.PrimalAndDualFeasible:
                status_str = "feasible"
                primal     = M.primalObjValue()
                dual       = M.dualObjValue()

            elif status in [ProblemStatus.DualInfeasible,
                            ProblemStatus.PrimalInfeasible]:
                status_str = "infeasible"
            elif status == ProblemStatus.Unknown:
                status_str = "unknown"
                code, desc = mosek.Env.getcodedesc(
                    mosek.rescode(int(M.getSolverIntInfo("optimizeResponse"))))
                error_message = f"Termination code {code}: {desc}"
                if verbose > 0:
                    print("The solution status is unknown.\n" + error_message)
                return {"status": status_str, "error": error_message}
            else:
                status_str = "other"
                error_message = ("Another unexpected problem, " +
                                 f"status {status} has been obtained.")
                print(error_message)
                return {"status": status_str, "error": error_message}
        except OptimizeError as e:
            status_str = "other"
            error_message = f"Optimization failed. Error: {e}"
            print(error_message)
            return {"status": status_str, "error": error_message}
        except SolutionError as e:
            status_str = M.getProblemStatus()
            error_message = f"Solution status: {status_str}. Error: {e}"
            print(error_message)
            return {"status": status_str, "error": error_message}
        except Exception as e:
            status_str = "other"
            error_message = "Unexpected error: {0}".format(e)
            print(error_message)
            return {"status": status_str, "error": error_message}

        if status_str in ["feasible", "infeasible"]:
            certificate = {x: 0 for x in known_vars}

            # c0(P(a...|x...))
            if not feas_as_optim:
                for x in set(objective).intersection(known_vars):
                    certificate[x] += objective[x]

            # + Tr[Z*F0(P(a...|x...))]=\sum_i*x_{kn_i}(P(a...|x...))*F_{kn_i}
            if mask_matrices:
                for x in set(Fi).intersection(known_vars):
                    support = Fi[x].nonzero()
                    certificate[x] = np.dot(ymat[support], Fi[x][support].A[0])

            # + I · b
            if inequalities:
                Ivalues = I.level() if solve_dual else -ineq_constraint.dual()
                for i, inequality in enumerate(inequalities):
                    for x in set(inequality).intersection(known_vars):
                        certificate[x] += Ivalues[i] * inequality[x]

            # + E · d
            if var_equalities:
                Evalues = E.level() if solve_dual else -eq_constraint.dual()
                for i, equality in enumerate(var_equalities):
                    for x in set(equality).intersection(known_vars):
                        certificate[x] += Evalues[i] * equality[x]

            # Clean entries with coefficient zero
            for x in list(certificate.keys()):
                if np.isclose(certificate[x], 0):
                    del certificate[x]

            # For debugging purposes
            if status_str == "feasible" and verbose > 1:
                TOL = 1e-8  # Constraint tolerance
                if inequalities:
                    x = A.todense().A \
                        @ np.array(list(x_values.values())) \
                        + b.T.A[0]
                    if np.any(x < -TOL):
                        print("Warning: Inequality constraints not satisfied" +
                              f" to {TOL} precision.")
                        print("Inequality constraints and their deviation:")
                        print([(ineq, x[i]) for i, (violated, ineq)
                               in enumerate(zip(x < -TOL, inequalities))
                               if violated])
                if var_equalities:
                    x = C.todense().A \
                        @ np.array(list(x_values.values())) \
                        + d.T.A[0]
                    if np.any(np.abs(x) > TOL):
                        print("Warning: Equality constraints not satisfied " +
                              f"to {TOL} precision.")
                        print("Equality constraints and their deviation:")
                        print([(eq, x[i]) for i, (violated, eq)
                               in enumerate(zip(np.abs(x) > TOL,
                                                var_equalities))
                               if violated])

            vars_of_interest = {"primal_value": primal, "dual_value": dual,
                                "status": status_str, "F": xmat, "Z": ymat,
                                "dual_certificate": certificate,
                                "x": x_values}

            return vars_of_interest
        else:
            return {"status": status_str}
