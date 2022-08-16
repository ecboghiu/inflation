import numpy as np
import scipy.sparse
import sys
from typing import Dict, Tuple
from warnings import warn
from time import time


def solveSDP_MosekFUSION(positionsmatrix: scipy.sparse.lil_matrix,
                         objective={'1': 0}, known_vars={},
                         semiknown_vars={}, positive_vars=[],
                         verbose=0, feas_as_optim=False, solverparameters={},
                         var_lowerbounds={}, var_upperbounds={},
                         var_inequalities=[], var_equalities=[],
                         solve_dual=True) -> Tuple[Dict, float, str]:
    """Internal function to solve the SDP with the MOSEK Fusion API.

    Now follows an extended description of how the SDP is encoded. In general,
    it is prefered to solve using the dual formulation, which is the default.

    The primal is written as follows:

    max c0 + c·x                                      Eq.(1)
    s.t. F=F0 + \sum Fi xi >= 0

    F0 is the constant entries of the moment matrix, and Fi is the matrix
    storing whose entry (n,m) stores the value of the coefficient of the
    moment xi at position (n,m) in the moment matrix.

    The dual of Eq (1) is:

    min c0 + Tr Z F0                                  Eq.(2)
    s.t. Tr Z Fi = - ci for all i
         Z >= 0

    Typically, all the probability information is stored in F0, and the
    coefficients Fi don't depend on the probabilties. However, if we
    use LPI constraints, then Fi can depend on the probability. The form of
    the SDP does not change, the Fi are updated to reflect that
    some coefficients are fixed by the probability.

    If we have a constant objective function, then we have a feasibility
    problem. It can be relaxed to:

    max lambda                                         Eq.(3)
    s.t. F=F0 + \sum Fi xi - lambda \Id >= 0

    with dual:

    min Tr Z F0                                        Eq.(4)
    s.t. Tr Z Fi = 0 for all i
            Z >= 0, Tr Z = 1

    This still allows for the extraction of certificates. If we use a Z_P1
    obtained from running problem (4) on probability P1, and we find
    Tr Z_P1 F0[P2] < 0, then clearly this is an upper bound of (4), and thus
    we can certifiy that the optimisation when using P2 will be negative.

    If we have upper and lower bounds on the variables, plus matrix
    equalities and inequalities, the problem changes as follows:

    max c0 + c·x                                        Eq.(5)
    s.t. F = F0 + \sum Fi xi >= 0,
         xi - l_i >= 0 for all i
         u_i - xi >= 0 for all i
         A x - b >= 0
         C x - d = 0
         c0 is a scalar
         c, x, b, d are vectors
         F0, Fi, A, C are matrices

    with dual:

    min Tr Z F0 - L·l + U·u + I·b + E·d                            Eq.(6)
    s.t. Tr Z Fi + ci - Ui + Li + A^T I + C^T E = 0 for all i
         Z >= 0, L >= 0, U >= 0, I >= 0, E unbounded
         Z is a matrix
         L, U, I, E are vectors

    Relaxed feasibility problems simply add the constraint Tr Z = 1.:

    min Tr Z F0 - L·l + U·u + I·b + E·d                            Eq.(7)
    s.t. Tr Z Fi + ci - Ui + Li + A^T I + C^T E = 0 for all i
         Z >= 0 (Mat), L >= 0 (Vec), U >= 0 (Vec), I >= 0 (Vec), E unbounded (Vec)
         Tr Z = 1

    Note that if we use values extracted from a minimisation over a distribution P1,
    Z_P1, L_P1, U_P1, I_P1, E_P1, and evaluate the expression on a distribution
    P2 and find that it is negative,

    Tr Z_P1 F0[P2] - L_P1·l + U_P1·u + I_P1·b + E_P1·d <= 0,

    this again upper bounds the global optimum of (7), so we can certify that
    P2 would lead to an infeasible unrelaxed feasibility problem.


    Parameters
    ----------
    positionsmatrix : scipy.sparse.lil_matrix
        Matrix of positions of the monomials in the moment matrix.
    objective : dict, optional
        Dictionary with keys as monomials and as values the monomial's
        coefficient in the objective function, by default {1: 0.}
    known_vars : dict, optional
        Dictionary of values for monomials (keys), by default {0: 0., 1: 1.}
    semiknown_vars : dict, optional
        Dictionary encoding proportionality constraints between
        different monomials, by default {}.
    positive_vars : list, optional
        List of monomials that are positive, by default []. Internally,
        for all variables in postive_vars, we override the lower bound
        to be 0 if it is negative or not set.
    var_lowerbounds : dict, optional
        Dictionary of lower bounds for monomials, by default {}.
    var_upperbounds : dict, optional
        Dictionary of upper bounds for monomials, by default {}.
    var_inequalities : list, optional
        List of inequalities enconded as dictionaries of coefficients,
        by default [].
    var_equalities : list, optional
        List of equalities enconded as dictionaries of coefficients,
        by default [].
    solve_dual : bool, optional
        Whether to solve the dual (True) or primal (False) formulation,
        by default True.
    verbose : int, optional
        How much information to display to the user, by default 0.
    feas_as_optim : bool, optional
        Whether to treat feasibility problems, where the objective is,
        constant, as an optimisation problem, by default False.
    solverparameters : dict, optional
        Dictionary of parameters to pass to the MOSEK solver, see
        https://docs.mosek.com/latest/pythonfusion/solver-parameters.html,
        by default {}.

    Returns
    -------
    Tuple[Dict,float,str]
        The first element of the tuple is a dictionary containing the
        optimisation information such as the 1) primal objective value,
        2) the moment matrix, 3) the dual values, 4) the certificate and
        a 5) dictionary of values for the monomials, in the following keys in
        the same order: 1) 'sol', 2) 'G', 3) 'Z', 4) 'dual_certificate',
        5) 'xi'. The second element is the objective value and the last is
        the problem status.
    """

    import mosek
    from mosek.fusion import Matrix, Model, ObjectiveSense, Expr, Domain, \
        OptimizeError, SolutionError, \
        AccSolutionStatus, ProblemStatus

    if verbose > 1:
        t0 = time()
        print('Starting pre-processing for the SDP solver.')

    CONSTANT_KEY = '1'  # If we hash monomials differently, we might
    # need to change the hash of the constant/offset term
    # from the number 1 to something else.
    # The rest of the code should be insensitive to
    # the hash except this line (ideally).

    # If the user didn't add the constant term.
    for equality in var_equalities:
        if CONSTANT_KEY not in equality:
            equality[CONSTANT_KEY] = 0
    for inequality in var_inequalities:
        if CONSTANT_KEY not in inequality:
            inequality[CONSTANT_KEY] = 0

    # variables = set(positionsmatrix.flatten())
    # positionsmatrix = positionsmatrix.astype(np.uint16)
    # mat_dim = positionsmatrix.shape[0]
    # F0 = scipy.sparse.lil_matrix((mat_dim,mat_dim))
    # Fi = {}
    # for x in variables:
    #     coeffmat = scipy.sparse.lil_matrix((mat_dim,mat_dim))
    #     coeffmat[scipy.sparse.find(positionsmatrix == x)[:2]] = 1
    #     Fi[x] = coeffmat

    Fi = positionsmatrix.copy()
    Fi = {k: scipy.sparse.lil_matrix(v, dtype=np.float) for k, v in Fi.items()}
    variables = set(list(Fi.keys()))
    mat_dim = Fi[next(iter(Fi))].shape[0]
    F0 = scipy.sparse.lil_matrix((mat_dim, mat_dim))

    # For positive variables, override the lower bound to be 0 if it is smaller
    for x in positive_vars:
        try:
            if var_lowerbounds[x] < 0:
                var_lowerbounds[x] = 0
        except KeyError:
            var_lowerbounds[x] = 0

    # Remove variables that are fixed by known_vars from the list of
    # variables, and also remove the corresponding entries for its upper
    # and lower bounds.
    for x, xval in known_vars.items():
        F0 += xval * Fi[x]
        variables.remove(x)
        # We do not delete Fi[x] because we need them later for the certificate.

        # Now update the bounds for known variables.
        if x in var_lowerbounds:
            if var_lowerbounds[x] >= xval:
                # We warn the user when these are incompatible, but the
                # program will continue.
                # TODO Should we remove this check? It is unlikely that
                # the bounds will be incompatible with fixed values, if the
                # user uses our program correctly.
                UserWarning(
                    "Lower bound {} for variable {}".format(var_lowerbounds[x], x) +
                    " is incompatible with the known value {}.".format(xval) +
                    " The lower bound will be ignored.")
            del var_lowerbounds[x]
        if x in var_upperbounds:
            if var_upperbounds[x] <= xval:
                UserWarning(
                    "Upper bound {} for variable {}".format(var_upperbounds[x], x) +
                    " is incompatible with the known value {}.".format(xval) +
                    " The upper bound will be ignored.")
            del var_upperbounds[x]
        for equality in var_equalities:
            if x != CONSTANT_KEY and x in equality:
                equality[CONSTANT_KEY] += equality[x] * xval
                del equality[x]
        for inequality in var_inequalities:
            if x != CONSTANT_KEY and x in inequality:
                inequality[CONSTANT_KEY] += inequality[x] * xval
                del inequality[x]

    # If some variable is semiknown, then this is the same as a constraint
    # of the form x_i = a_i * x_j. Imposing this constraint is equivalent
    # to adding to the coeffmat of x_j  the expression a_i * (coeffmat of x_i),
    # and removing the coeffmat of x_i. Note: this also accounts for the case
    # where x_j is not found anywhere else in the moment matrix.
    for x, (c, x2) in semiknown_vars.items():
        Fi[x2] += c * Fi[x]
        variables.remove(x)

        del Fi[x]  # We can safely delete Fi[x].

        # TODO Is worthile to consider the compatibility of lower and upper
        # bounds of variables involved in LPI constraints? I would say no.
        # For our usecase, it should not happen that we have one upper bound
        # for one variable and lower bound for the other variable such that
        # the constraint can never be satisfied, but as a general NPO package,
        # this might happen.
        if x in var_lowerbounds and x2 in var_lowerbounds:
            del var_lowerbounds[x]
        if x in var_upperbounds and x2 in var_upperbounds:
            del var_upperbounds[x]
        for equality in var_equalities:
            if x in equality:
                try:
                    equality[x2] += c * equality[x]
                except KeyError:
                    equality[x2] = c * equality[x]
                del equality[x]
        for inequality in var_inequalities:
            if x in inequality:
                try:
                    inequality[x2] += c * inequality[x]
                except KeyError:
                    inequality[x2] = c * inequality[x]
                del inequality[x]

    var2index = {x: i for i, x in enumerate(variables)}

    # Calculate the matrices A, C and vectors b, d such that
    # Ax + b >= 0, Cx + d == 0.
    # Comment: it appears that storing sparse matrices as dictionaries of
    # indicies (dok format) is the best way to build matrices incrementally
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html

    b = scipy.sparse.dok_matrix((len(var_inequalities), 1))
    for i, inequality in enumerate(var_inequalities):
        b[i, 0] = inequality[CONSTANT_KEY]

    A = scipy.sparse.dok_matrix((len(var_inequalities), len(variables)))
    for i, inequality in enumerate(var_inequalities):
        for x, c in inequality.items():
            if x != CONSTANT_KEY:
                A[i, var2index[x]] = c

    d = scipy.sparse.dok_matrix((len(var_equalities), 1))
    for i, equality in enumerate(var_equalities):
        d[i, 0] = equality[CONSTANT_KEY]

    C = scipy.sparse.dok_matrix((len(var_equalities), len(variables)))
    for i, equality in enumerate(var_equalities):
        for x, c in equality.items():
            if x != CONSTANT_KEY:
                C[i, var2index[x]] = c

    # Before converting to MOSEK format, it is useful to keep indices of where
    # F0, Fi are nonzero, as it seems to be more difficult to extract later.
    ij_F0_nonzero = [(i, j) for (i, j) in zip(*F0.nonzero()) if j >= i]
    ij_Fi_nonzero = {x: [(i, j) for (i, j) in zip(*Fi[x].nonzero()) if j >= i]
                     for x in variables}

    # Convert to MOSEK format.
    b = Matrix.sparse(*b.shape, *b.nonzero(), b[b.nonzero()].A[0])
    # A = Matrix.sparse(*A.shape, *A.nonzero(), A[A.nonzero()].A[0])
    d = Matrix.sparse(*d.shape, *d.nonzero(), d[d.nonzero()].A[0])
    # C = Matrix.sparse(*C.shape, *C.nonzero(), C[C.nonzero()].A[0])
    F0 = Matrix.sparse(*F0.shape, *F0.nonzero(), F0[F0.nonzero()].A[0])
    for x in variables:
        F = Fi[x]
        Fi[x] = Matrix.sparse(*F.shape, *F.nonzero(), F[F.nonzero()].A[0])

    # Find if the objective is constant.
    # We can safely assume that with the known_constraints, all known variables
    # in the objective have already been removed, and added to
    # objective[CONSTANT_KEY]. However, just to be certain, we will
    # do this step also here.
    for x, val in known_vars.items():
        if x in objective and x != CONSTANT_KEY:
            objective[CONSTANT_KEY] += val * objective[x]
            del objective[x]
    constant_objective = False
    if list(objective.keys()) == [CONSTANT_KEY]:
        constant_objective = True
        if verbose > 1:
            print('Constant objective detected! Treating the problem as ' +
                  'a feasibility problem.')

    if var_lowerbounds:
        # The following dictionary, lowerbounded_var2idx, is because the L
        # dual variable will only be defined for the variables with explicitly
        # defined lower bounds (this is to avoid having to write a -infinity
        # lower bound). As such, it is useful to have a dictionary that maps
        # a monomial x to the index of i of Li, where Li is the dual variable
        # of the constraint x - lb_x >= 0.
        lowerbounded_var2idx = {x: i for i, x in enumerate(var_lowerbounds)}
    if var_upperbounds:
        # Same as above.
        upperbounded_var2idx = {x: i for i, x in enumerate(var_upperbounds)}

    if verbose > 1:
        print('Pre-processing took', time() - t0, "seconds.")
        t0 = time()
        print("Building the model.")

    M = Model('InflationSDP')

    if solve_dual:
        # Define variables
        Z = M.variable(Domain.inPSDCone(mat_dim))
        if var_lowerbounds:
            L = M.variable(len(var_lowerbounds), Domain.greaterThan(0))
        if var_upperbounds:
            U = M.variable(len(var_upperbounds), Domain.greaterThan(0))
        if var_inequalities:
            I = M.variable(len(var_inequalities), Domain.greaterThan(0))
            # The following is calculating for all i the expression
            # \sum_j I_j A_ji as it will later be useful for the constraints
            # It seems MOSEK Fusion API does not allow me to pick index i
            # of an expression (A^T I)_i, so I have to do it manually.
            # I slice the columns of A and dot them with I and store it in
            # a list, AtI
            A = A.tocsr()  # csr for the fastest row slicing
            # however there might be even better ways
            # see https://stackoverflow.com/questions/13843352/what-is-the-fastest-way-to-slice-a-scipy-sparse-matrix
            AtI = []  # \sum_j I_j A_ji as i-th entry of AtI
            for i in range(len(variables)):
                slicee = A[:, i]
                slicee_moseksparse = Matrix.sparse(*slicee.shape,
                                                   *slicee.nonzero(),
                                                   slicee[slicee.nonzero()].A[0])
                AtI.append(Expr.dot(slicee_moseksparse, I))
        if var_equalities:
            E = M.variable(len(var_equalities), Domain.unbounded())
            C = C.tocsr()
            CtI = []  # \sum_j E_j C_ji as i-th entry of CtI
            for i in range(len(variables)):
                slicee = C[:, i]
                slicee_moseksparse = Matrix.sparse(*slicee.shape,
                                                   *slicee.nonzero(),
                                                   slicee[slicee.nonzero()].A[0])
                CtI.append(Expr.dot(slicee_moseksparse, E))

        # Define objective function
        mosek_obj = Expr.dot(Z, F0)
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
        if not constant_objective and not feas_as_optim:
            # If we are doing a relaxed feasibility test, then we want the
            # interpretation of the result being the maximum minimum eigenvalue.
            # However, if we add a constant to the objective, then this
            # interpretation breaks down, so we only add it if the objective
            # is not constant.
            mosek_obj = Expr.add(mosek_obj, float(objective[CONSTANT_KEY]))

        # Set objective function
        # c0 + Tr Z F0 - L·lb + U·ub + I·b + E·d
        M.objective(ObjectiveSense.Minimize, mosek_obj)

        # Add constraints
        ci_constraints = []
        for i, x in enumerate(variables):
            lhs = Expr.dot(Z, Fi[x])
            ci = objective[x] if x in objective else 0
            lhs = Expr.add(lhs, float(ci))
            if x in var_upperbounds:
                lhs = Expr.sub(lhs, U.index(upperbounded_var2idx[x]))
            if x in var_lowerbounds:
                lhs = Expr.add(lhs, L.index(lowerbounded_var2idx[x]))
            if var_inequalities:
                lhs = Expr.add(lhs, AtI[i])
            if var_equalities:
                lhs = Expr.sub(lhs, CtI[i])

            # Tr Z Fi + ci - Ui + Li + \sum_j I_j A_ji + \sum_j E_j A_ji = 0
            ci_constraints.append(M.constraint(lhs, Domain.equalsTo(0)))

        if feas_as_optim:
            # Tr Z = 1
            ci_constraints.append(M.constraint(Expr.dot(Z, Matrix.eye(mat_dim)),
                                               Domain.equalsTo(1)))
    else:
        # Define variables
        x_mosek = M.variable(len(variables), Domain.unbounded())

        # Add upper and lower bounds
        if var_lowerbounds:
            lb_constraints = []
            for x, val in var_lowerbounds.items():
                try:
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
            ineq_constraint = M.constraint(Expr.add(Expr.mul(A_mosek, x_mosek), b),
                                           Domain.greaterThan(0))
        if var_equalities:
            C_mosek = Matrix.sparse(*C.shape, *C.nonzero(), C[C.nonzero()].A[0])
            eq_constraint = M.constraint(Expr.add(Expr.mul(C_mosek, x_mosek), d),
                                         Domain.equalsTo(0))

        G = M.variable("G", Domain.inPSDCone(mat_dim))
        if constant_objective and feas_as_optim:
            lam = M.variable(1, Domain.unbounded())

        #### Adding constraints
        ### Comment 1:
        # I found no better way than defining a positive variable G
        # and imposing that G - F0 - \sum_i xi Fi = 0 *elementwise*.
        # It seems the MOSEK Fusion API does not allow for the multiplication
        # of a constant matrix by a scalar variable, xi * Fi, which
        # leads to many small equality constraints. This is one reason
        # why the dual formulation seems preferable.
        # TODO Reinvestigate whether using .pick or something similar
        # can lead to a more efficient formulation of the primal.
        ### Comment 2 on loop performance:
        # The approach taken below:
        # 1. loop over all (i, j>=i) once and store G[i,j] in a constraints mat.
        # 2. loop over all xi, and for all (i,j) in the support of Fi
        # substract in the constraint matrix Expr.mul(xi,Fi(i,j)).
        # 3. loop again over all (i, j>=i) and add to the model that all
        # constraints after step 2. in the constraints. mat have domain 0.
        # There could be more efficient ways to do this, which can be
        # implemented in the future.
        constraints = np.empty((mat_dim, mat_dim), dtype=object)
        for i in range(mat_dim):
            for j in range(i, mat_dim):
                constraints[i, j] = G.index(i, j)
        for i, j in ij_F0_nonzero:
            constraints[i, j] = Expr.sub(constraints[i, j], F0.get(i, j))
        for i, xi in enumerate(variables):
            for i_, j_ in ij_Fi_nonzero[xi]:
                constraints[i_, j_] = Expr.sub(constraints[i_, j_],
                                               Expr.mul(Fi[xi].get(i_, j_),
                                                        x_mosek.index(i)))
        if constant_objective and feas_as_optim:
            for i in range(mat_dim):
                # lambda on the diagonal
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
                mosek_obj = float(objective[CONSTANT_KEY])
        else:
            mosek_obj = float(objective[CONSTANT_KEY])
            for xi, ci in objective.items():
                if xi != CONSTANT_KEY:
                    mosek_obj = Expr.add(mosek_obj,
                                         Expr.mul(float(ci),
                                                  x_mosek.index(var2index[xi])))

        M.objective(ObjectiveSense.Maximize, mosek_obj)

    if verbose > 1:
        print('Built the model in', time() - t0, 'seconds.')
        t0 = time()
        print("Solving the model.")

    xmat, ymat, primal, dual = None, None, None, None
    xmat = np.zeros((mat_dim, mat_dim))
    try:
        if verbose > 0:
            M.setLogHandler(sys.stdout)

        if solverparameters:
            for param, val in solverparameters.items():
                M.setSolverParam(param, val)

        M.acceptedSolutionStatus(AccSolutionStatus.Anything)

        M.solve()

        if solve_dual:
            ymat = Z.level().reshape([mat_dim, mat_dim])
            xmat = F0.getDataAsArray().reshape((mat_dim, mat_dim))
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


        elif (status == ProblemStatus.DualInfeasible or
              status == ProblemStatus.PrimalInfeasible):
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
            print("Another unexpected problem status {0} is obtained.".format(status))
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
        # c0
        certificate = {CONSTANT_KEY: objective[CONSTANT_KEY]}
        if constant_objective and feas_as_optim:
            certificate[CONSTANT_KEY] = 0  # For feas as optim, we don't want the offset c0

        # += Tr Z F0 = \sum_i x_{known i} * F_{known i}}
        for x in known_vars:
            support = Fi[x].nonzero()
            certificate[x] = np.dot(ymat[support], Fi[x][support].A[0])
        if var_lowerbounds:
            # += - L · lb
            Lvalues = L.level() if solve_dual else [-c.dual() for c in lb_constraints]
            for i, (x, lb) in enumerate(var_lowerbounds.items()):
                certificate[CONSTANT_KEY] -= Lvalues[i] * lb
        if var_upperbounds:
            # += U · ub
            Uvalues = U.level() if solve_dual else [-c.dual() for c in ub_constraints]
            for i, (x, ub) in enumerate(var_upperbounds.items()):
                certificate[CONSTANT_KEY] += Uvalues[i] * ub
        if var_inequalities:
            # += I · b
            Ivalues = I.level() if solve_dual else -ineq_constraint.dual()
            certificate[CONSTANT_KEY] += Ivalues @ b.getDataAsArray()
        if var_equalities:
            # += E · b
            Evalues = E.level() if solve_dual else -eq_constraint.dual()
            certificate[CONSTANT_KEY] += Evalues @ d.getDataAsArray()

        # If feasible, make sure the bounds are satisfied!
        # TODO Maybe remove this check in the future if we find this
        # error never appears
        if status_str == 'feasible':
            NEGATIVE_TOL = -1e-8  # How tolerant we are with negative values
            try:
                for x, lb in var_lowerbounds.items():
                    assert x_values[x] - lb >= NEGATIVE_TOL, "Lower bound violated."
                for x, ub in var_upperbounds.items():
                    assert ub - x_values[x] >= NEGATIVE_TOL, "Upper bound violated."
                if var_inequalities:
                    assert np.all(A @ np.array(list(x_values.values())) +
                                  b.getDataAsArray() >= NEGATIVE_TOL), "Inequalities not satisftied."
                if var_equalities:
                    assert np.allclose(C @ np.array(list(x_values.values())) +
                                       d.getDataAsArray(), 0), "Equalities not satisfied by the solution."
            except AssertionError as e:
                print("Error: The solution does not satisfy some constraints. {}".format(e))
                return None, None, None

        vars_of_interest = {'sol': primal, 'G': xmat, 'Z': ymat,
                            'dual_certificate': certificate,
                            'x': x_values}

        return vars_of_interest, primal, status_str
    else:
        return None, None, status_str

def solveSDP_MosekFUSION2(positionsmatrix: scipy.sparse.lil_matrix,
                         objective={1: 0.}, known_vars={0: 0., 1: 1.},
                         semiknown_vars={}, positive_vars=[],
                         verbose=0, feas_as_optim=False, solverparameters={},
                         var_lowerbounds={}, var_upperbounds={},
                         var_inequalities=[], var_equalities=[],
                         solve_dual=True) -> Tuple[Dict, float, str]:
    """Internal function to solve the SDP with the MOSEK Fusion API.

    Now follows an extended description of how the SDP is encoded. In general,
    it is prefered to solve using the dual formulation, which is the default.

    The primal is written as follows:

    max c0 + c·x                                      Eq.(1)
    s.t. F=F0 + \sum Fi xi >= 0

    F0 is the constant entries of the moment matrix, and Fi is the matrix
    storing whose entry (n,m) stores the value of the coefficient of the
    moment xi at position (n,m) in the moment matrix.

    The dual of Eq (1) is:

    min c0 + Tr Z F0                                  Eq.(2)
    s.t. Tr Z Fi = - ci for all i
         Z >= 0

    Typically, all the probability information is stored in F0, and the
    coefficients Fi don't depend on the probabilties. However, if we
    use LPI constraints, then Fi can depend on the probability. The form of
    the SDP does not change, the Fi are updated to reflect that
    some coefficients are fixed by the probability.

    If we have a constant objective function, then we have a feasibility
    problem. It can be relaxed to:

    max lambda                                         Eq.(3)
    s.t. F=F0 + \sum Fi xi - lambda \Id >= 0

    with dual:

    min Tr Z F0                                        Eq.(4)
    s.t. Tr Z Fi = 0 for all i
            Z >= 0, Tr Z = 1

    This still allows for the extraction of certificates. If we use a Z_P1
    obtained from running problem (4) on probability P1, and we find
    Tr Z_P1 F0[P2] < 0, then clearly this is an upper bound of (4), and thus
    we can certifiy that the optimisation when using P2 will be negative.

    If we have upper and lower bounds on the variables, the problems
    changes as follows:

    max c0 + c·x                                        Eq.(5)
    s.t. F = F0 + \sum Fi xi >= 0,
         xi - l_i >=0,
         u_i - xi >= 0

    with dual:

    min Tr Z F0 - L·l + U·u                             Eq.(6)
    s.t. Tr Z Fi = - ci + Ui - Li for all i
         Z >= 0, L >= 0, U >= 0

    Relaxed feasibility problems change as follows when we have upper and
    lower bounds:

    min Tr Z F0 - L·l + U·u                             Eq.(7)
    s.t. Tr Z Fi = Ui - Li for all i
         Z >= 0, L >= 0, U >= 0, Tr Z = 1

    The unrelaxed version simply lacks the Tr Z = 1 constraint. Note that
    if we use values extracted from a minimisation over a distribution P1,
    Z_P1, L_P1, U_P1 and evaluate the expression on a distribution P2 and
    find that it is negative,

    Tr Z_P1 F0[P2] - L_P1·l + U_P1·u <= 0

    This again upper bounds the global optimum of (7), so we can certify that
    P2 would lead to an infeasible unrelaxed feasibility problem. We simply
    need to add a constant shift to the certificate.


    Parameters
    ----------
    positionsmatrix : scipy.sparse.lil_matrix
        Matrix of positions of the monomials in the moment matrix.
    objective : dict, optional
        Dictionary with keys as monomials and as values the monomial's
        coefficient in the objective function, by default {1: 0.}
    known_vars : dict, optional
        Dictionary of values for monomials (keys), by default {0: 0., 1: 1.}
    semiknown_vars : dict, optional
        Dictionary encoding proportionality constraints between
        different monomials, by default {}.
    positive_vars : list, optional
        List of monomials that are positive, by default []. Internally,
        for all variables in postive_vars, we override the lower bound
        to be 0 if it is negative or not set.
    var_lowerbounds : dict, optional
        Dictionary of lower bounds for monomials, by default {}.
    var_upperbounds : dict, optional
        Dictionary of upper bounds for monomials, by default {}.
    var_inequalities : list, optional
        List of inequalities encoded as dictionaries of coefficients,
        by default [].
    var_equalities : list, optional
        List of equalities encoded as dictionaries of coefficients,
        by default [].
    solve_dual : bool, optional
        Whether to solve the dual (True) or primal (False) formulation,
        by default True.
    verbose : int, optional
        How much information to display to the user, by default 0.
    feas_as_optim : bool, optional
        Whether to treat feasibility problems, where the objective is,
        constant, as an optimisation problem, by default False.
    solverparameters : dict, optional
        Dictionary of parameters to pass to the MOSEK solver, see
        https://docs.mosek.com/latest/pythonfusion/solver-parameters.html,
        by default {}.

    Returns
    -------
    Tuple[Dict,float,str]
        The first element of the tuple is a dictionary containing the
        optimisation information such as the 1) primal objective value,
        2) the moment matrix, 3) the dual values, 4) the certificate and
        a 5) dictionary of values for the monomials, in the following keys in
        the same order: 1) 'sol', 2) 'G', 3) 'Z', 4) 'dual_certificate',
        5) 'xi'. The second element is the objective value and the last is
        the problem status.
    """

    # TODO Make sure we have the correct sign when extracting constraint duals
    # TODO If we want to allow for subnormalised moment matrices, we need
    # to decouple the constant variable from the moment '1'. Currently in the
    # objective value, 1 is reserved for constants. However, this is not
    # relevant for current applications.
    # TODO remove all the upper and lower bounds, as they can be ab
    # into the inequalities matrix.

    import mosek
    from mosek.fusion import Matrix, Model, ObjectiveSense, Expr, Domain, \
                             OptimizeError, SolutionError, \
                             AccSolutionStatus, ProblemStatus

    # Temporary fix for compatiblity with the rest
    try:
        del known_vars[0]  # TODO Fix a convention
    except KeyError:
        pass

    if verbose > 1:
        t0 = time()
        print('Starting pre-processing for the SDP solver.')

    positionsmatrix = positionsmatrix.astype(np.uint16)
    mat_dim = positionsmatrix.shape[0]

    CONSTANT_KEY = 1      # If we hash monomials differently, we might
                          # need to change the hash of the constant/offset term
                          # from the number 1 to something else.
                          # The rest of the code should be insensitive to
                          # the hash except this line (ideally).

    # If the user didn't add the constant term.
    for equality in var_equalities:
        if CONSTANT_KEY not in equality:
            equality[CONSTANT_KEY] = 0
    for inequality in var_inequalities:
        if CONSTANT_KEY not in inequality:
            inequality[CONSTANT_KEY] = 0

    variables = set(positionsmatrix.flatten())
    variables.discard(0)

    F0 = scipy.sparse.lil_matrix((mat_dim,mat_dim))
    Fi = {}
    for x in variables:
        coeffmat = scipy.sparse.lil_matrix((mat_dim,mat_dim))
        coeffmat[scipy.sparse.find(positionsmatrix == x)[:2]] = 1
        Fi[x] = coeffmat

    # ! (From here onwards we no longer need positionsmatrix for anything.)

    # For positive variables, override the lower bound to be 0 if it is smaller
    # print("Positive vars:", positive_vars)
    for x in positive_vars:
        try:
            if var_lowerbounds[x] < 0:
                var_lowerbounds[x] = 0
        except KeyError:
            if x>0:
                var_lowerbounds[x] = 0

    # Remove variables that are fixed by known_vars from the list of
    # variables, and also remove the corresponding entries for its upper
    # and lower bounds.
    for x, xval in known_vars.items():
        F0 += xval * Fi[x]
        variables.remove(x)
        # We do not delete Fi[x] because we need them later for the certificate.

        # Now update the bounds for known variables.
        if x in var_lowerbounds:
            if var_lowerbounds[x] >= xval:
                # We warn the user when these are incompatible, but the
                # program will continue.
                # TODO Should we remove this check? It is unlikely that
                # the bounds will be incompatible with fixed values, if the
                # user uses our program correctly.
                UserWarning(
                    "Lower bound {} for variable {}".format(var_lowerbounds[x], x) +
                    " is incompatible with the known value {}.".format(xval) +
                    " The lower bound will be ignored.")
            del var_lowerbounds[x]
        if x in var_upperbounds:
            if var_upperbounds[x] <= xval:
                    UserWarning(
                    "Upper bound {} for variable {}".format(var_upperbounds[x], x) +
                    " is incompatible with the known value {}.".format(xval) +
                    " The upper bound will be ignored.")
            del var_upperbounds[x]
        for equality in var_equalities:
            if x != CONSTANT_KEY and x in equality:
                equality[CONSTANT_KEY] += equality[x] * xval
                del equality[x]
        for inequality in var_inequalities:
            if x != CONSTANT_KEY and x in inequality:
                inequality[CONSTANT_KEY] += inequality[x] * xval
                del inequality[x]

    # If some variable is semiknown, then this is the same as a constraint
    # of the form x_i = a_i * x_j. Imposing this constraint is equivalent
    # to adding to the coeffmat of x_j  the expression a_i * (coeffmat of x_i),
    # and removing the coeffmat of x_i. Note: this also accounts for the case
    # where x_j is not found anywhere else in the moment matrix.
    for x, (c, x2) in semiknown_vars.items():
        Fi[x2] += c * Fi[x]
        variables.remove(x)

        del Fi[x] # We can safely delete Fi[x].

        # TODO Is worthile to consider the compatibility of lower and upper
        # bounds of variables involved in LPI constraints? I would say no.
        # For our usecase, it should not happen that we have one upper bound
        # for one variable and lower bound for the other variable such that
        # the constraint can never be satisfied, but as a general NPO package,
        # this might happen.
        if x in var_lowerbounds and x2 in var_lowerbounds:
            del var_lowerbounds[x]
        if x in var_upperbounds and x2 in var_upperbounds:
            del var_upperbounds[x]
        for equality in var_equalities:
            if x in equality:
                try:
                    equality[x2] += c * equality[x]
                except KeyError:
                    equality[x2] = c * equality[x]
                del equality[x]
        for inequality in var_inequalities:
            if x in inequality:
                try:
                    inequality[x2] += c * inequality[x]
                except KeyError:
                    inequality[x2] = c * inequality[x]
                del inequality[x]

    var2index = {x: i for i, x in enumerate(variables)}

    # Calculate the matrices A, C and vectors b, d such that
    # Ax + b >= 0, Cx + d == 0.
    # Comment: it appears that storing sparse matrices as dictionaries of
    # indicies (dok format) is the best way to build matrices incrementally
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html

    b = scipy.sparse.dok_matrix((len(var_inequalities), 1))
    for i, inequality in enumerate(var_inequalities):
        b[i, 0] = inequality[CONSTANT_KEY]

    A = scipy.sparse.dok_matrix((len(var_inequalities), len(variables)))
    for i, inequality in enumerate(var_inequalities):
        for x, c in inequality.items():
            if x != CONSTANT_KEY:
                A[i, var2index[x]] = c

    d = scipy.sparse.dok_matrix((len(var_equalities), 1))
    for i, equality in enumerate(var_equalities):
        d[i, 0] = equality[CONSTANT_KEY]

    C = scipy.sparse.dok_matrix((len(var_equalities), len(variables)))
    for i, equality in enumerate(var_equalities):
        for x, c in equality.items():
            if x != CONSTANT_KEY:
                C[i, var2index[x]] = c


    # Before converting to MOSEK format, it is useful to keep indices of where
    # F0, Fi are nonzero, as it seems to be more difficult to extract later.
    ij_F0_nonzero =      [(i,j) for (i,j) in zip(*F0.nonzero())    if j >= i]
    ij_Fi_nonzero = {x : [(i,j) for (i,j) in zip(*Fi[x].nonzero()) if j >= i]
                                                            for x in variables}

    # Convert to MOSEK format.
    b = Matrix.sparse(*b.shape, *b.nonzero(), b[b.nonzero()].A[0])
    # A = Matrix.sparse(*A.shape, *A.nonzero(), A[A.nonzero()].A[0])
    d = Matrix.sparse(*d.shape, *d.nonzero(), d[d.nonzero()].A[0])
    # C = Matrix.sparse(*C.shape, *C.nonzero(), C[C.nonzero()].A[0])
    F0 = Matrix.sparse(*F0.shape, *F0.nonzero(), F0[F0.nonzero()].A[0])
    for x in variables:
        F = Fi[x]
        Fi[x] = Matrix.sparse(*F.shape, *F.nonzero(), F[F.nonzero()].A[0])

    # Find if the objective is constant.
    # We can safely assume that with the known_constraints, all known variables
    # in the objective have already been removed, and added to
    # objective[CONSTANT_KEY]. However, just to be certain, we will
    # do this step also here.
    for x, val in known_vars.items():
        if x in objective and x != CONSTANT_KEY:
            objective[CONSTANT_KEY] += val * objective[x]
            del objective[x]
    constant_objective = False
    if list(objective.keys()) == [CONSTANT_KEY]:
        constant_objective = True
        if verbose > 1:
            print('Constant objective detected! Treating the problem as ' +
                  'a feasibility problem.')

    if var_lowerbounds:
        # The following dictionary, lowerbounded_var2idx, is because the L
        # dual variable will only be defined for the variables with explicitly
        # defined lower bounds (this is to avoid having to write a -infinity
        # lower bound). As such, it is useful to have a dictionary that maps
        # a monomial x to the index of i of Li, where Li is the dual variable
        # of the constraint x - lb_x >= 0.
        lowerbounded_var2idx = {x: i for i, x in enumerate(var_lowerbounds)}
    if var_upperbounds:
        # Same as above.
        upperbounded_var2idx = {x: i for i, x in enumerate(var_upperbounds)}

    if verbose > 1:
        print('Pre-processing took', time() - t0, "seconds.")
        t0 = time()
        print("Building the model.")

    M = Model('InflationSDP')

    if solve_dual:
        # Define variables
        Z = M.variable(Domain.inPSDCone(mat_dim))
        if var_lowerbounds:
            L = M.variable(len(var_lowerbounds), Domain.greaterThan(0))
        if var_upperbounds:
            U = M.variable(len(var_upperbounds), Domain.greaterThan(0))
        if var_inequalities:
            I = M.variable(len(var_inequalities), Domain.greaterThan(0))
            # The following is calculating for all i the expression
            # \sum_j I_j A_ji as it will later be useful for the constraints
            # It seems MOSEK Fusion API does not allow me to pick index i
            # of an expression (A^T I)_i, so I have to do it manually.
            # I slice the columns of A and dot them with I and store it in
            # a list, AtI
            A = A.tocsr()   # csr for the fastest row slicing
                            # however there might be even better ways
                            # see https://stackoverflow.com/questions/13843352/what-is-the-fastest-way-to-slice-a-scipy-sparse-matrix
            AtI = []  # \sum_j I_j A_ji as i-th entry of AtI
            for i in range(len(variables)):
                slicee = A[:, i]
                slicee_moseksparse = Matrix.sparse(*slicee.shape,
                                                   *slicee.nonzero(),
                                                   slicee[slicee.nonzero()].A[0])
                AtI.append(Expr.dot(slicee_moseksparse, I))
        if var_equalities:
            E = M.variable(len(var_equalities), Domain.unbounded())
            C = C.tocsr()
            CtI = []  # \sum_j E_j C_ji as i-th entry of CtI
            for i in range(len(variables)):
                slicee = C[:, i]
                slicee_moseksparse = Matrix.sparse(*slicee.shape,
                                                   *slicee.nonzero(),
                                                   slicee[slicee.nonzero()].A[0])
                CtI.append(Expr.dot(slicee_moseksparse, E))

        # Define objective function
        mosek_obj = Expr.dot(Z, F0)
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
        if not constant_objective and not feas_as_optim:
            # If we are doing a relaxed feasibility test, then we want the
            # interpretation of the result being the maximum minimum eigenvalue.
            # However, if we add a constant to the objective, then this
            # interpretation breaks down, so we only add it if the objective
            # is not constant.
            mosek_obj = Expr.add(mosek_obj, float(objective[CONSTANT_KEY]))

        # Set objective function
        # c0 + Tr Z F0 - L·lb + U·ub + I·b + E·d
        M.objective(ObjectiveSense.Minimize, mosek_obj)

        # Add constraints
        ci_constraints = []
        for i, x in enumerate(variables):
            lhs = Expr.dot(Z, Fi[x])
            ci = objective[x] if x in objective else 0
            lhs = Expr.add(lhs, float(ci))
            if x in var_upperbounds:
                lhs = Expr.sub(lhs, U.index(upperbounded_var2idx[x]))
            if x in var_lowerbounds:
                lhs = Expr.add(lhs, L.index(lowerbounded_var2idx[x]))
            if var_inequalities:
                lhs = Expr.add(lhs, AtI[i])
            if var_equalities:
                lhs = Expr.sub(lhs, CtI[i])

            # Tr Z Fi + ci - Ui + Li + \sum_j I_j A_ji + \sum_j E_j A_ji = 0
            ci_constraints.append(M.constraint(lhs, Domain.equalsTo(0)))

        if feas_as_optim:
            # Tr Z = 1
            ci_constraints.append(M.constraint(Expr.dot(Z, Matrix.eye(mat_dim)),
                                               Domain.equalsTo(1)))
    else:
        # Define variables
        x_mosek = M.variable(len(variables), Domain.unbounded())

        # Add upper and lower bounds
        if var_lowerbounds:
            lb_constraints = []
            for x, val in var_lowerbounds.items():
                try:
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
            ineq_constraint = M.constraint(Expr.add(Expr.mul(A_mosek,x_mosek), b),
                                           Domain.greaterThan(0))
        if var_equalities:
            C_mosek = Matrix.sparse(*C.shape, *C.nonzero(), C[C.nonzero()].A[0])
            eq_constraint = M.constraint(Expr.add(Expr.mul(C_mosek,x_mosek), d),
                                         Domain.equalsTo(0))

        G = M.variable("G", Domain.inPSDCone(mat_dim))
        if constant_objective and feas_as_optim:
            lam = M.variable(1, Domain.unbounded())

        #### Adding constraints
        ### Comment 1:
        # I found no better way than defining a positive variable G
        # and imposing that G - F0 - \sum_i xi Fi = 0 *elementwise*.
        # It seems the MOSEK Fusion API does not allow for the multiplication
        # of a constant matrix by a scalar variable, xi * Fi, which
        # leads to many small equality constraints. This is one reason
        # why the dual formulation seems preferable.
        # TODO Reinvestigate whether using .pick or something similar
        # can lead to a more efficient formulation of the primal.
        ### Comment 2 on loop performance:
        # The approach taken below:
        # 1. loop over all (i, j>=i) once and store G[i,j] in a constraints mat.
        # 2. loop over all xi, and for all (i,j) in the support of Fi
        # substract in the constraint matrix Expr.mul(xi,Fi(i,j)).
        # 3. loop again over all (i, j>=i) and add to the model that all
        # constraints after step 2. in the constraints. mat have domain 0.
        # There could be more efficient ways to do this, which can be
        # implemented in the future.
        constraints = np.empty((mat_dim,mat_dim), dtype=object)
        for i in range(mat_dim):
            for j in range(i, mat_dim):
                constraints[i, j] = G.index(i, j)
        for i, j in ij_F0_nonzero:
            constraints[i ,j] = Expr.sub(constraints[i, j], F0.get(i, j))
        for i, xi in enumerate(variables):
            for i_, j_ in ij_Fi_nonzero[xi]:
                constraints[i_, j_] = Expr.sub(constraints[i_, j_],
                                             Expr.mul(Fi[xi].get(i_, j_),
                                                      x_mosek.index(i)))
        if constant_objective and feas_as_optim:
            for i in range(mat_dim):
                # lambda on the diagonal
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
                mosek_obj = float(objective[CONSTANT_KEY])
        else:
            mosek_obj = float(objective[CONSTANT_KEY])
            for xi, ci in objective.items():
                if xi != CONSTANT_KEY:
                    mosek_obj = Expr.add(mosek_obj,
                                        Expr.mul(float(ci),
                                                x_mosek.index(var2index[xi])))

        M.objective(ObjectiveSense.Maximize, mosek_obj)

    if verbose > 1:
        print('Built the model in', time() - t0, 'seconds.')
        t0 = time()
        print("Solving the model.")


    xmat, ymat, primal, dual = None, None, None, None
    xmat = np.zeros((mat_dim,mat_dim))
    try:
        if verbose > 0:
            M.setLogHandler(sys.stdout)

        if solverparameters:
            for param, val in solverparameters.items():
                M.setSolverParam(param, val)

        M.acceptedSolutionStatus(AccSolutionStatus.Anything)

        M.solve()

        if solve_dual:
            ymat = Z.level().reshape([mat_dim, mat_dim])
            xmat = F0.getDataAsArray().reshape((mat_dim,mat_dim))
            x_values = {}
            for i, x in enumerate(variables):
                x_values[x] = -ci_constraints[i].dual()[0]
                xmat = xmat + (x_values[x] *
                               Fi[x].getDataAsArray().reshape((mat_dim,mat_dim)))
        else:
            ymat = G.dual().reshape([mat_dim, mat_dim])
            xmat = G.level().reshape([mat_dim, mat_dim])
            x_values = {x: val for x, val in zip(variables, x_mosek.level())}

        status = M.getProblemStatus()
        if status == ProblemStatus.PrimalAndDualFeasible:
            status_str = 'feasible'
            primal = M.primalObjValue()
            dual = M.dualObjValue()


        elif (status == ProblemStatus.DualInfeasible or
              status == ProblemStatus.PrimalInfeasible):
            status_str = 'infeasible'
        elif status == ProblemStatus.Unknown:
            status_str = 'unknown'
            # The solutions status is unknown. The termination code
            # indicates why the optimizer terminated prematurely.
            print("The solution status is unknown.")
            symname, desc = mosek.Env.getcodedesc(
                    mosek.rescode(int(M.getSolverIntInfo("optimizeResponse"))))
            print("   Termination code: {0} {1}".format(symname, desc))
        else:
            status_str = 'other'
            print("Another unexpected problem status {0} is obtained.".format(status))
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
        # c0
        certificate = {CONSTANT_KEY: objective[CONSTANT_KEY]}
        if constant_objective and feas_as_optim:
            certificate[CONSTANT_KEY] = 0  # For feas as optim, we don't want the offset c0

        # += Tr Z F0 = \sum_i x_{known i} * F_{known i}}
        for x in known_vars:
            support = Fi[x].nonzero()
            certificate[x] = np.dot(ymat[support], Fi[x][support].A[0])
        if var_lowerbounds:
            # += - L · lb
            Lvalues = L.level() if solve_dual else [-c.dual() for c in lb_constraints]
            for i, (x, lb) in enumerate(var_lowerbounds.items()):
                certificate[CONSTANT_KEY] -= Lvalues[i] * lb
        if var_upperbounds:
            # += U · ub
            Uvalues = U.level() if solve_dual else [-c.dual() for c in ub_constraints]
            for i, (x, ub) in enumerate(var_upperbounds.items()):
                certificate[CONSTANT_KEY] += Uvalues[i] * ub
        if var_inequalities:
            # += I · b
            Ivalues = I.level() if solve_dual else -ineq_constraint.dual()
            certificate[CONSTANT_KEY] += Ivalues @ b.getDataAsArray()
        if var_equalities:
            # += E · b
            Evalues = E.level() if solve_dual else -eq_constraint.dual()
            certificate[CONSTANT_KEY] += Evalues @ d.getDataAsArray()

        # If feasible, make sure the bounds are satisfied!
        # TODO Maybe remove this check in the future if we find this
        # error never appears
        if status_str == 'feasible':
            NEGATIVE_TOL = -1e-8 # How tolerant we are with negative values
            try:
                for x, lb in var_lowerbounds.items():
                    assert x_values[x] - lb >= NEGATIVE_TOL, "Lower bound violated."
                for x, ub in var_upperbounds.items():
                    assert ub - x_values[x] >= NEGATIVE_TOL, "Upper bound violated."
                if var_inequalities:
                    assert np.all(A @ np.array(list(x_values.values())) +
                                b.getDataAsArray() >= NEGATIVE_TOL), "Inequalities not satisfied."
                if var_equalities:
                    assert np.allclose(C @ np.array(list(x_values.values())) +
                                d.getDataAsArray(), 0), "Equalities not satisfied by the solution."
            except AssertionError as e:
                print("Error: The solution does not satisfy some constraints. {}".format(e))
                return None, None, None

        vars_of_interest = {'sol': primal, 'G': xmat, 'Z': ymat,
                            'dual_certificate': certificate,
                            'x': x_values}

        return vars_of_interest, primal, status_str
    else:
        return None, None, status_str