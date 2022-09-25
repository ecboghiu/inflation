"""
This file contains the functions to send the problems to SDP solvers.
@authors: Alejandro Pozas-Kerstjens, Emanuel-Cristian Boghiu and Elie Wolfe
"""
import numpy as np
import sys

from scipy.sparse import lil_matrix, dok_matrix
from typing import Dict, Tuple


def solveSDP_MosekFUSION(mask_matrices= {},
                         objective={'1': 0.},
                         known_vars={'0': 0., '1': 1.},
                         semiknown_vars={},
                         positive_vars=[],
                         verbose=0,
                         feas_as_optim=False,
                         solverparameters={},
                         var_lowerbounds={},
                         var_upperbounds={},
                         var_inequalities=[],
                         var_equalities=[],
                         solve_dual=True,
                         process_constraints=True
                         ) -> Tuple[Dict, float, str]:
    r"""Internal function to solve the SDP with the `MOSEK Fusion API
    <https://docs.mosek.com/latest/pythonfusion/index.html>`_.

    Now follows an extended description of how the SDP is encoded. In general,
    it is prefered to solve using the dual formulation, which is the default.

    The primal is written as follows:

    .. math::

        \text{max}\quad & c_0 + c\cdot x\\
        \text{s.t.}\quad & F_0 + \sum F_i x_i \succeq 0

    :math:`F_0` is the constant entries of the moment matrix, and :math:`F_i` is
    the matrix whose entry :math:`(n,m)` stores the value of the coefficient of
    the moment :math:`x_i` at position :math:`(n,m)` in the moment matrix.

    The dual of the equation above is:

    .. math::

        \text{min}\quad & c_0+\text{Tr}(Z\cdot F_0)\\
        \text{s.t.}\quad & \text{Tr}(Z\cdot F_i) = - c_i \,\forall\, i,\\
        &Z \succeq 0.

    Typically, all the probability information is stored in :math:`F_0`, and the
    coefficients :math:`F_i` do not depend on the probabilities. However, if we
    use LPI constraints (see, e.g., `arXiv:2203.16543
    <http://www.arxiv.org/abs/2203.16543/>`_), then :math:`F_i` can depend on
    the probabilities. The form of the SDP does not change, in any case.

    If we have a constant objective function, then we have a feasibility
    problem. It can be rewritten into the following optimization problem:

    .. math::
        \text{max}\quad&\lambda\\
        \text{s.t.}\quad& F_0 + \sum F_i x_i - \lambda \cdot 1 \succeq 0,

    which achieves :math:`\lambda\geq 0` if the original problem is feasible and
    :math:`\lambda<0` otherwise. The dual of this problem is:

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
        \text{s.t.}\quad & \text{Tr}(Z \cdot F_i) = -c_i+U_i-L_i \,\forall\,i,\\
        & Z \succeq 0,\,L \geq 0,\,U \geq 0.

    The relaxed feasibility problems change accordingly.

    Parameters
    ----------
    maskmatrices_name_dict : dict
        A dictionary with keys as monomials and values as scipy sparse arrays
        indicating the locations of the monomial in the moment matrix.
    objective : dict, optional
        Dictionary with keys as monomials and as values the monomial's
        coefficient in the objective function. By default ``{1: 0.}``
    known_vars : dict, optional
        Dictionary of values for monomials (keys). By default ``{0: 0., 1: 1.}``
    semiknown_vars : dict, optional
        Dictionary encoding proportionality constraints between
        different monomials. By default ``{}``.
    var_lowerbounds : dict, optional
        Dictionary of lower bounds for monomials. By default ``{}``.
    var_upperbounds : dict, optional
        Dictionary of upper bounds for monomials. By default ``{}``.
    var_inequalities : list, optional
        List of inequalities encoded as dictionaries of coefficients. By
        default ``[]``.
    var_equalities : list, optional
        List of equalities encoded as dictionaries of coefficients. By default
        ``[]``.
    solve_dual : bool, optional
        Whether to solve the dual (True) or primal (False) formulation. By
        default ``True``.
    verbose : int, optional
        How much information to display to the user. By default ``0``.
    feas_as_optim : bool, optional
        Whether to treat feasibility problems, where the objective is,
        constant, as an optimisation problem. By default ``False``.
    solverparameters : dict, optional
        Dictionary of parameters to pass to the MOSEK solver, see `MOSEK's
        documentation
        <https://docs.mosek.com/latest/pythonfusion/solver-parameters.html>`_.
        By default ``{}``.

    Returns
    -------
    Tuple[Dict, float, str]
        The first element of the tuple is a dictionary containing the
        optimisation information such as the 1) primal objective value,
        2) the moment matrix, 3) the dual values, 4) the certificate and
        a 5) dictionary of values for the monomials, in the following keys in
        the same order: 1) ``sol``, 2) ``G``, 3) ``Z``, 4) ``dual_certificate``,
        5) ``xi``. The second element is the objective value and the last is the
        problem status.
    """
    import mosek
    from mosek.fusion import Matrix, Model, ObjectiveSense, Expr, Domain, \
        OptimizeError, SolutionError, \
        AccSolutionStatus, ProblemStatus

    if verbose > 1:
        from time import perf_counter
        t0 = perf_counter()
        print('Starting pre-processing for the SDP solver.')
    
    Fi = mask_matrices
    Fi = {k: lil_matrix(v, dtype=float) for k, v in Fi.items()}
    mat_dim   = Fi[next(iter(Fi))].shape[0]  # We assume all are the same size
    variables = set(list(Fi.keys()))
    
    CONST_KEY = 'CONST_KEY'  
    
    assert CONST_KEY not in Fi, f"{CONST_KEY} is a reserved key."
    Fi[CONST_KEY] = 0
    assert CONST_KEY not in objective, f"{CONST_KEY} is a reserved key."
    objective[CONST_KEY] = 0
    for equality in var_equalities:
        assert CONST_KEY not in equality, f"{CONST_KEY} is a reserved key."
        equality[CONST_KEY] = 0
    for inequality in var_inequalities:
        assert CONST_KEY not in inequality, f"{CONST_KEY} is a reserved key."
        inequality[CONST_KEY] = 0

    
    #F0        = lil_matrix((mat_dim, mat_dim))

    # # Sanity check
    # if verbose > 1:
    #     for eq in var_equalities:
    #         for var in eq:
    #             assert var in variables, f"Variable {var} in equality {eq} not in variables."
    #     for ineq in var_inequalities:
    #         for var in ineq:
    #             assert var in variables, f"Variable {var} in equality {eq} not in variables."

    # if process_constraints:
    #     # For positive variables, override the lower bound to be 0 if it is smaller
    #     for x in positive_vars:
    #         try:
    #             if var_lowerbounds[x] < 0:
    #                 var_lowerbounds[x] = 0
    #         except KeyError:
    #             var_lowerbounds[x] = 0

    # Remove variables that are fixed by known_vars from the list of
    # variables, and also remove the corresponding entries in the constraints
    for x, xval in known_vars.items():
        Fi[CONST_KEY] += xval * Fi[x]
        variables.remove(x)
        # We do not delete Fi[x] because we need them later for the certificate.
        # Now update the bounds for known variables.
        # if x in var_lowerbounds:
        #     if var_lowerbounds[x] >= xval:
        #         # We warn the user when these are incompatible, but the
        #         # program will continue.
        #         UserWarning(
        #             "Lower bound {} for variable {}".format(var_lowerbounds[x], x) +
        #             " is incompatible with the known value {}.".format(xval) +
        #             " The lower bound will be ignored.")
        #     del var_lowerbounds[x]
        # if x in var_upperbounds:
        #     if var_upperbounds[x] <= xval:
        #         UserWarning(
        #             "Upper bound {} for variable {}".format(var_upperbounds[x], x) +
        #             " is incompatible with the known value {}.".format(xval) +
        #             " The upper bound will be ignored.")
        #     del var_upperbounds[x]
        if x in var_lowerbounds:
            del var_lowerbounds[x]
        if x in var_upperbounds:
            del var_upperbounds[x]
        if x in objective:
            objective[CONST_KEY] += xval * objective[x]
        for equality in var_equalities:
            if x in equality:
                equality[CONST_KEY] += xval * equality[x] 
        for inequality in var_inequalities:
            if x in inequality:
                inequality[CONST_KEY] += xval * inequality[x]

    # for equality in var_equalities:
    #     for x, xval in known_vars.items():
    #         equality[CONST_KEY] += xval * equality[x] 
    # for inequality in var_inequalities:
    #     for x, xval in known_vars.items():
    #         inequality[CONST_KEY] += xval * inequality[x] 

    constant_objective = False
    if list(objective.keys()) == [CONST_KEY]:
        constant_objective = True
        if verbose > 1:
            print('Constant objective detected. Treating the problem as ' +
                  'a feasibility problem.')

    if process_constraints:
        # If some variable is semiknown, then this is the same as a constraint
        # of the form x_i = a_i * x_j. Imposing this constraint is equivalent
        # to adding to the coeffmat of x_j  the expression a_i * (coeffmat of x_i),
        # and removing the coeffmat of x_i. Note: this also accounts for the case
        # where x_j is not found anywhere else in the moment matrix.
        for x, (c, x2) in semiknown_vars.items():
            Fi[x2] += c * Fi[x]
            variables.remove(x)

            del Fi[x]  # We can safely delete Fi[x].

            # TODO Is worthwhile to consider the compatibility of lower and upper
            # bounds of variables involved in LPI constraints? I would say no.
            # For our usecase, it should not happen that we have one upper bound
            # for one variable and lower bound for the other variable such that
            # the constraint can never be satisfied, but as a general NPO package,
            # this might happen.
            if x in var_lowerbounds and x2 in var_lowerbounds:
                del var_lowerbounds[x]
            if x in var_upperbounds and x2 in var_upperbounds:
                del var_upperbounds[x]  # TODO can i remove this
            if x in objective:
                objective[x2] = objective.get(x2, 0) + c * objective[x]
            for equality in var_equalities:
                if x in equality:
                    equality[x2] = equality.get(x2, 0) + c * equality[x]
            for inequality in var_inequalities:
                if x in inequality:
                    inequality[x2] = inequality.get(x2, 0) + c * inequality[x]
    else:
        # If we do not process the semi-known constraints, we add them as 
        # equality constraints.
        for x, (c, x2) in semiknown_vars.items():
            var_equalities.append({x: 1, x2: -c})

    var2index = {x: i for i, x in enumerate(variables)}

    # Calculate the matrices A, C and vectors b, d such that
    # Ax + b >= 0, Cx + d == 0.
    b = dok_matrix((len(var_inequalities), 1))
    for i, inequality in enumerate(var_inequalities):
        b[i, 0] = inequality[CONST_KEY]

    A = dok_matrix((len(var_inequalities), len(variables)))
    for i, inequality in enumerate(var_inequalities):
        for x, c in inequality.items():
            if x in variables:
                A[i, var2index[x]] = c

    d = dok_matrix((len(var_equalities), 1))
    for i, equality in enumerate(var_equalities):
        d[i, 0] = equality[CONST_KEY]

    C = dok_matrix((len(var_equalities), len(variables)))
    for i, equality in enumerate(var_equalities):
        for x, c in equality.items():
            if x in variables:
                C[i, var2index[x]] = c

    # Before converting to MOSEK format, it is useful to keep indices of where
    # F0, Fi are nonzero, as it seems to be more difficult to extract later.
    ij_F0_nonzero = [(i, j) for (i, j) in zip(*Fi[CONST_KEY].nonzero()) if j >= i]
    ij_Fi_nonzero = {x: [(i, j) for (i, j) in zip(*Fi[x].nonzero()) if j >= i]
                     for x in variables}

    # Convert to MOSEK format.
    b = Matrix.sparse(*b.shape, *b.nonzero(), b[b.nonzero()].A[0])
    # A = Matrix.sparse(*A.shape, *A.nonzero(), A[A.nonzero()].A[0])
    d = Matrix.sparse(*d.shape, *d.nonzero(), d[d.nonzero()].A[0])
    # C = Matrix.sparse(*C.shape, *C.nonzero(), C[C.nonzero()].A[0])
    F0 = Fi[CONST_KEY]
    Fi[CONST_KEY] = Matrix.sparse(*F0.shape, *F0.nonzero(), F0[F0.nonzero()].A[0])
    for x in variables:
        F = Fi[x]
        Fi[x] = Matrix.sparse(*F.shape, *F.nonzero(), F[F.nonzero()].A[0])


    if var_lowerbounds:
        lowerbounded_var2idx = {x: i for i, x in enumerate(var_lowerbounds)}
    if var_upperbounds:
        upperbounded_var2idx = {x: i for i, x in enumerate(var_upperbounds)}

    if verbose > 1:
        print('Pre-processing took', format(perf_counter() - t0, ".4f"), "seconds.")
        t0 = perf_counter()
    if verbose > 0:
        print("Building the model...")

    M = Model('InflationSDP')

    if solve_dual:
        # Define variables
        Z = M.variable('Z', Domain.inPSDCone(mat_dim))
        if var_lowerbounds:
            L = M.variable('L', len(var_lowerbounds), Domain.greaterThan(0))
        if var_upperbounds:
            U = M.variable('U', len(var_upperbounds), Domain.greaterThan(0))
        if var_inequalities:
            I = M.variable('I', len(var_inequalities), Domain.greaterThan(0))
            # It seems MOSEK Fusion API does not allow to pick index i
            # of an expression (A^T I)_i, so this does it manually row by row.
            A = A.tocsr()
            AtI = []  # \sum_j I_j A_ji as i-th entry of AtI
            for i in range(len(variables)):
                slice_ = A[:, i]
                slice_moseksparse = Matrix.sparse(*slice_.shape,
                                                  *slice_.nonzero(),
                                                  slice_[slice_.nonzero()].A[0])
                AtI.append(Expr.dot(slice_moseksparse, I))
        if var_equalities:
            E = M.variable('E', len(var_equalities), Domain.unbounded())
            C = C.tocsr()
            CtI = []  # \sum_j E_j C_ji as i-th entry of CtI
            for i in range(len(variables)):
                slice_ = C[:, i]
                slice_moseksparse = Matrix.sparse(*slice_.shape,
                                                  *slice_.nonzero(),
                                                  slice_[slice_.nonzero()].A[0])
                CtI.append(Expr.dot(slice_moseksparse, E))

        # Define and set objective function
        # Tr Z F0 - L·lb + U·ub + I·b + E·d + c0
        mosek_obj = Expr.dot(Z, Fi[CONST_KEY])
        if var_lowerbounds:
            mosek_obj = Expr.sub(mosek_obj,
                                 Expr.dot(L, list(var_lowerbounds.values())))
        if var_upperbounds:
            mosek_obj = Expr.add(mosek_obj,
                                 Expr.dot(U, list(var_upperbounds.values())))
        if var_inequalities:
            mosek_obj = Expr.add(mosek_obj, Expr.dot(I, b))
        if var_equalities:
            mosek_obj = Expr.add(mosek_obj, Expr.dot(E, d))
        if not feas_as_optim:
            mosek_obj = Expr.add(mosek_obj, float(objective[CONST_KEY]))

        M.objective(ObjectiveSense.Minimize, mosek_obj)

        # Add constraints
        # Tr Z Fi + ci - Ui + Li + \sum_j I_j A_ji + \sum_j E_j A_ji == 0
        ci_constraints = []
        for i, x in enumerate(variables):
            lhs = Expr.dot(Z, Fi[x])
            ci  = objective[x] if x in objective else 0
            lhs = Expr.add(lhs, float(ci))
            if x in var_upperbounds:
                lhs = Expr.sub(lhs, U.index(upperbounded_var2idx[x]))
            if x in var_lowerbounds:
                lhs = Expr.add(lhs, L.index(lowerbounded_var2idx[x]))
            if var_inequalities:
                lhs = Expr.add(lhs, AtI[i])
            if var_equalities:
                lhs = Expr.add(lhs, CtI[i])

            ci_constraints.append(M.constraint(f'c{i}', lhs, Domain.equalsTo(0)))

        if feas_as_optim:
            # When solving a feasibility problem as max t st M + t*1 >= 0, we
            # have an extra Fi = 1, so we add the corresponding constraint,
            # Tr Z = 1.
            ci_constraints.append(M.constraint('trZ=1',
                                               Expr.dot(Z, Matrix.eye(mat_dim)),
                                               Domain.equalsTo(1)))
    else:
        # Set up the problem in primal formulation
        # Define variables
        x_mosek = M.variable('x', len(variables), Domain.unbounded())

        # Add upper and lower bounds
        if var_lowerbounds:
            lb_constraints = []
            for x, val in var_lowerbounds.items():
                try:  # TODO get rid of try except?
                    # x_i - lb_i >= 0
                    lb_constraints.append(M.constraint(
                        Expr.sub(x_mosek.index(var2index[x]),
                                 float(val)),
                        Domain.greaterThan(0)))
                except KeyError:
                    pass
        if var_upperbounds:
            ub_constraints = []
            for x, val in var_upperbounds.items():
                try:
                    # ub_i - x_i >= 0
                    ub_constraints.append(M.constraint(
                        Expr.sub(float(val),
                                 x_mosek.index(var2index[x])),
                        Domain.greaterThan(0)))
                except KeyError:
                    pass

        if var_inequalities:
            A_mosek = Matrix.sparse(*A.shape, *A.nonzero(), A[A.nonzero()].A[0])
            ineq_constraint = M.constraint('Ineq', Expr.add(Expr.mul(A_mosek, x_mosek),
                                                    b),
                                           Domain.greaterThan(0))
        if var_equalities:
            C_mosek = Matrix.sparse(*C.shape, *C.nonzero(), C[C.nonzero()].A[0])
            eq_constraint = M.constraint('Eq', Expr.add(Expr.mul(C_mosek, x_mosek), d),
                                         Domain.equalsTo(0))

        G = M.variable("G", Domain.inPSDCone(mat_dim))
        if constant_objective and feas_as_optim:
            lam = M.variable('lam', 1, Domain.unbounded())

        # Add constraints
        constraints = np.empty((mat_dim, mat_dim), dtype=object)
        for i in range(mat_dim):
            for j in range(i, mat_dim):
                constraints[i, j] = G.index(i, j)
        for i, j in ij_F0_nonzero:
            constraints[i, j] = Expr.sub(constraints[i, j], Fi[CONST_KEY].get(i, j))
        for i, xi in enumerate(variables):
            for i_, j_ in ij_Fi_nonzero[xi]:
                constraints[i_, j_] = Expr.sub(constraints[i_, j_],
                                             Expr.mul(Fi[xi].get(i_, j_),
                                                      x_mosek.index(i)))
        if constant_objective and feas_as_optim:
            for i in range(mat_dim):
                # Put lam on the diagonal
                constraints[i, i] = Expr.add(constraints[i, i], lam)
        for i in range(mat_dim):
            for j in range(i, mat_dim):
                # G(i,j) - F0(i,j) - sum_i xi Fi(i,j) = 0
                M.constraint(constraints[i, j], Domain.equalsTo(0))

        # Set objective function
        if constant_objective:
            if feas_as_optim:
                mosek_obj = lam
            else:
                mosek_obj = float(objective[CONST_KEY])
        else:
            mosek_obj = float(objective[CONST_KEY])
            for xi, ci in objective.items():
                if xi != CONST_KEY:
                    mosek_obj = Expr.add(mosek_obj,
                                         Expr.mul(float(ci),
                                                  x_mosek.index(var2index[xi])))

        M.objective(ObjectiveSense.Maximize, mosek_obj)

    if verbose > 1:
        print('Model built in', format(perf_counter() - t0, ".4f"), 'seconds.')
        M.writeTask('InflationSDPModel.ptf')
        print('Model saved to InflationSDPModel.ptf.')
        t0 = perf_counter()
    if verbose > 0:
        print("Solving the model...")

    # Solving and readout
    xmat, ymat, primal, dual = None, None, None, None
    try:
        if verbose > 1:
            M.setLogHandler(sys.stdout)

        if solverparameters:
            for param, val in solverparameters.items():
                M.setSolverParam(param, val)

        M.acceptedSolutionStatus(AccSolutionStatus.Anything)

        M.solve()

        if solve_dual:
            ymat = Z.level().reshape([mat_dim, mat_dim])
            xmat = Fi[CONST_KEY].getDataAsArray().reshape((mat_dim, mat_dim))
            x_values = {}
            for i, x in enumerate(variables):
                x_values[x] = -ci_constraints[i].dual()[0]
                xmat = xmat + (x_values[x] *
                             Fi[x].getDataAsArray().reshape((mat_dim, mat_dim)))
        else:
            ymat = G.dual().reshape([mat_dim, mat_dim])
            xmat = G.level().reshape([mat_dim, mat_dim])
            x_values = {x: val for x, val in zip(variables, x_mosek.level())}

        status = M.getProblemStatus()
        if status == ProblemStatus.PrimalAndDualFeasible:
            status_str = 'feasible'
            primal = M.primalObjValue()
            dual = M.dualObjValue()

        elif status in [ProblemStatus.DualInfeasible,
                        ProblemStatus.PrimalInfeasible]:
            status_str = 'infeasible'
        elif status == ProblemStatus.Unknown:
            status_str = 'unknown'
            # The solutions status is unknown. The termination code
            # indicates why the optimizer terminated prematurely.
            if verbose > 0:
                print("The solution status is unknown.")
                symname, desc = mosek.Env.getcodedesc(
                    mosek.rescode(int(M.getSolverIntInfo("optimizeResponse"))))
                print("   Termination code: {0} {1}".format(symname, desc))
        else:
            status_str = 'other'
            print("Another unexpected problem, status {0}".format(status) +
                  " has been obtained.")
    except OptimizeError as e:
        print("Optimization failed. Error: {0}".format(e))
        return None, None, None
    except SolutionError as e:
        status = M.getProblemStatus()
        print("Solution status: {}. Error: {0}".format(status, e))
        return None, None, status
    except Exception as e:
        print("Unexpected error: {0}".format(e))
        return None, None, None

    if status_str in ['feasible', 'infeasible']:
        certificate = {}
        
        # c0(P(a...|x...))
        if not feas_as_optim:
            for x in objective:
                if x in known_vars:
                    certificate[x] += objective[x]

        # + Tr Z F0(P(a...|x...)) = \sum_i x_{known i}(P(a...|x...))*F_{known i}
        for x in known_vars:
            support = Fi[x].nonzero()
            certificate[x] = np.dot(ymat[support], Fi[x][support].A[0])
                
        # - L · lb
        if var_lowerbounds:
            Lvalues = L.level() if solve_dual else [-c.dual() for c in lb_constraints]
            for i, (x, lb) in enumerate(var_lowerbounds.items()):
                certificate[CONST_KEY] -= Lvalues[i] * lb
                
        # + U · ub
        if var_upperbounds:
            Uvalues = U.level() if solve_dual else [-c.dual() for c in ub_constraints]
            for i, (x, ub) in enumerate(var_upperbounds.items()):
                certificate[CONST_KEY] += Uvalues[i] * ub
                
        # + I · b
        if var_inequalities:
            Ivalues = I.level() if solve_dual else -ineq_constraint.dual()
            # certificate[CONST_KEY] += Ivalues @ b.getDataAsArray()
            for i, inequality in enumerate(var_inequalities):
                for x, coeff in inequality.items():
                    if x in known_vars:
                        certificate[x] += Evalues[i] * coeff

            for i, (x, coeff) in enumerate(var_inequalities.items()):
                if x in known_vars:
                    certificate[x] += Ivalues[i] * coeff
                    
        # + E · d
        if var_equalities:
            Evalues = E.level() if solve_dual else -eq_constraint.dual()
            # certificate[CONST_KEY] += Evalues @ d.getDataAsArray()
            for i, equality in enumerate(var_equalities):
                for x, coeff in equality.items():
                    if x in known_vars:
                        certificate[x] += Evalues[i] * coeff


        # For debugging purposes
        if status_str == 'feasible' and verbose > 1:
            TOL = 1e-8  # Constraint tolerance
            for x, lb in var_lowerbounds.items():
                if x in x_values:
                    if x_values[x] - lb <= -TOL:
                        print(f'Warning: Lower bound violated for {x} by {x_values[x] - lb}')
            for x, ub in var_upperbounds.items():
                if x in x_values:
                    if ub - x_values[x] <= -TOL:
                        print(f'Warning: Upper bound violated for {x} by {ub - x_values[x]}')
            if var_inequalities:
                x = (A.todense() @ np.array(list(x_values.values())) +
                                b.getDataAsArray()).A[0]
                if np.any(x < -TOL):
                    print(f'Warning: Inequality constraints not satisfied to {TOL} precision.')
                    print(f'Inequality constraints and their deviation from 0:')
                    print([(ineq, x[i]) for i, (violated, ineq) 
                           in enumerate(zip(x < -TOL, var_inequalities))
                           if violated])
            if var_equalities:
                x = (C.todense() @ np.array(list(x_values.values())) +
                                    d.getDataAsArray()).A[0]
                if np.any(np.abs(x) > TOL):
                    print(f'Warning: Equality constraints not satisfied to {TOL} precision.')
                    print(f'Equality constraints and their deviation from 0:')
                    print([(eq, x[i]) for i, (violated, eq) 
                           in enumerate(zip(np.abs(x) > TOL, var_equalities))
                           if violated])

        vars_of_interest = {'sol': primal, 'G': xmat, 'Z': ymat,
                            'dual_certificate': certificate,
                            'x': x_values}

        return vars_of_interest, primal, status_str
    else:
        return None, None, status_str
