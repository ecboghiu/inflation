import numpy as np
import scipy.sparse
import sys
from typing import Dict, Tuple
from warnings import warn

from sympy import var


def solveSDP_MosekFUSION(positionsmatrix: scipy.sparse.lil_matrix,
                         objective={1: 0.}, known_vars={0: 0., 1: 1.},
                         semiknown_vars={}, positive_vars=[],
                         verbose=0, feas_as_optim=False, solverparameters={}):
    import mosek
    from mosek.fusion import Matrix, Model, ObjectiveSense, Expr, Domain, \
                             OptimizeError, SolutionError, \
                             AccSolutionStatus, ProblemStatus

    solve_dual = True
    if solverparameters:
        solve_dual = True if 'solve_dual' not in solverparameters else solverparameters['solve_dual']

    use_positive_vars = False
    if len(positive_vars) > 0:
        use_positive_vars = True

    positionsmatrix = positionsmatrix.astype(np.uint16)
    if len(semiknown_vars) > 0:
        # When using proportionality constraints, we might get an entry that is
        # proportional to a variable that is not found at any other entry, thus
        # the total number of variables is encoded in semiknown_vars instead of
        # positionsmatrix.
        nr_variables = max(max([val[1] for val in semiknown_vars.values()]),
                           int(np.max(positionsmatrix))) + 1
    else:
        nr_variables = np.max(positionsmatrix) + 1
    nr_known        = len(known_vars)
    nr_unknown      = nr_variables - nr_known

    F0 = scipy.sparse.lil_matrix(positionsmatrix.shape)
    for var in known_vars.keys():
        F0[scipy.sparse.find(positionsmatrix == var)[:2]] = known_vars[var]
    F0 = Matrix.sparse(*F0.shape, *F0.nonzero(), F0[F0.nonzero()].todense().A[0])

    # List of empty sparse matrices
    Fi = []
    variables_order = []
    for variable in set(range(nr_variables)) - set(known_vars.keys()):
        # Set to 1 where the unknown variable is
        F = scipy.sparse.lil_matrix(positionsmatrix.shape)
        F[scipy.sparse.find(positionsmatrix == variable)[:2]] = 1
        Fi.append(F)
        variables_order.append(variable)

    if len(semiknown_vars) > 0:
        Fii = []
        for idx, var in enumerate(set(range(nr_variables))
                                  - set(known_vars.keys())):
            if var in semiknown_vars.keys():
                factor, subs      = semiknown_vars[var]
                Fi[subs-nr_known] += factor*Fi[var-nr_known]
            else:
                Fii.append(Fi[idx])
        Fi = Fii

        nr_variables -= len(semiknown_vars)
    # Convert to MOSEK format
    for i in range(len(Fi)):
        F     = scipy.sparse.lil_matrix(Fi[i])
        Fi[i] = Matrix.sparse(*F.shape,
                              *F.nonzero(),
                              F[F.nonzero()].todense().A[0])

    mat_dim = positionsmatrix.shape[0]

    # If the objective is a constant, treat it as a feasibility problem
    constant_objective = False
    if list(objective.keys()) == [1]:
        constant_objective = True
        if verbose > 1:
            print('Constant objective detected. Treating the problem as ' +
                  'a feasibility problem.')

    M = Model('InfSDP')
    if solve_dual:
        Z = M.variable("Z", Domain.inPSDCone(mat_dim))
        if not constant_objective:
            ci_constraints = []
            M.objective(ObjectiveSense.Minimize,
                        Expr.add(float(objective[1]),
                        Expr.dot(Z, F0)))
            for var, F in zip(variables_order, Fi):
                if var in objective.keys():
                    if use_positive_vars and (var in positive_vars):
                        ci_constraints.append(
                            M.constraint(Expr.dot(Z, F),
                                         Domain.lessThan(-float(objective[var])))
                                              )
                    else:
                        ci_constraints.append(
                            M.constraint(Expr.dot(Z, F),
                                         Domain.equalsTo(-float(objective[var])))
                                              )
                else:
                    ci_constraints.append(
                        M.constraint(Expr.dot(Z, F),
                                     Domain.equalsTo(0)))
        else:
            M.objective(ObjectiveSense.Minimize, Expr.dot(Z, F0))
            ci_constraints = []
            for var, F in zip(variables_order, Fi):
                # F = Fi[i]
                if use_positive_vars and (var in positive_vars):
                    ci_constraints.append(M.constraint(Expr.dot(Z, F),
                                                       Domain.lessThan(0)))
                else:
                    ci_constraints.append(M.constraint(Expr.dot(Z, F),
                                                       Domain.equalsTo(0)))
            if feas_as_optim:
                ci_constraints.append(
                    M.constraint(Expr.dot(Z, Matrix.eye(mat_dim)),
                                 Domain.equalsTo(1)))
    else:
        # The primal formulation uses a lot more RAM and is slower.
        # Only use if the problem is not too big
        x = M.variable(nr_variables, Domain.unbounded())
        for var, val in known_values.items():
            M.constraint(x.index(var), Domain.equalsTo(float(val)))

        if use_positive_vars:
            for i in positive_vars:
                M.constraint(x.index(i), Domain.greaterThan(0))

        G = M.variable("G", Domain.inPSDCone(mat_dim))
        if not feas_as_optim:
            M.objective(ObjectiveSense.Maximize, 0)
            for i in range(mat_dim):
                for j in range(i, mat_dim):
                    M.constraint(Expr.sub(G.index(i,j), x.index(int(positionsmatrix[i,j]))), Domain.equalsTo(0))
        else:
            if not constant_objective:
                c = np.zeros(nr_variables)
                vars_in_obj = np.array(sorted(list(objective.keys())), dtype=int)
                for var in vars_in_obj:
                    c[var] = float(objective[var])
                M.objective(ObjectiveSense.Maximize, Expr.dot(c, x))
                for i in range(mat_dim):
                    for j in range(i,mat_dim):
                        M.constraint(Expr.sub(G.index(i,j), x.index(int(positionsmatrix[i,j]))), Domain.equalsTo(0))
            else:
                lam = M.variable(1, Domain.unbounded())
                M.objective(ObjectiveSense.Maximize, lam)
                for i in range(mat_dim):
                    M.constraint(Expr.add(Expr.sub(G.index(i,i),
                                                   x.index(int(positionsmatrix[i,i]))),
                                          lam), Domain.equalsTo(0))
                    for j in range(i+1, mat_dim):
                        M.constraint(Expr.sub(G.index(i,j), x.index(int(positionsmatrix[i,j]))), Domain.equalsTo(0))

    xmat, ymat, primal, dual = None, None, None, None
    xmat = np.zeros((mat_dim,mat_dim))
    try:
        if verbose > 0:
            M.setLogHandler(sys.stdout)
        #M.setSolverParam("intpntCoTolInfeas", 1.0e-8)
        #M.setSolverParam("intpntCoTolPfeas", 1.0e-8)  # default 1e-8

        M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        M.solve()

        status = M.getProblemStatus()
        if status == ProblemStatus.PrimalAndDualFeasible:
            status_str = 'feasible'
            if solve_dual:
                ymat = Z.level().reshape([mat_dim, mat_dim])
                xmat = F0.getDataAsArray().reshape((mat_dim,mat_dim))
                xi_list = np.zeros(len(Fi))
                for i, F in enumerate(Fi):
                    xi_list[i] = -ci_constraints[i].dual()[0]
                    xmat = xmat + xi_list[i] * F.getDataAsArray().reshape((mat_dim,mat_dim))
            else:
                xmat = G.level().reshape([mat_dim, mat_dim])
                ymat = G.dual().reshape([mat_dim, mat_dim])
            primal = M.primalObjValue()
            dual = M.dualObjValue()
        elif status == ProblemStatus.DualInfeasible or status == ProblemStatus.PrimalInfeasible:
            status_str = 'infeasible'
            if solve_dual:
                ymat = Z.level().reshape([mat_dim, mat_dim])
                xmat = F0.getDataAsArray().reshape((mat_dim,mat_dim))
                xi_list = np.zeros(len(Fi))
                for i, F in enumerate(Fi):
                    xi_list[i] = -ci_constraints[i].dual()[0]
                    xmat = xmat + xi_list[i] * F.getDataAsArray().reshape((mat_dim,mat_dim))
            else:
                ymat = G.dual().reshape([mat_dim, mat_dim])
                xmat = G.level().reshape([mat_dim, mat_dim])
        elif status == ProblemStatus.Unknown:
            status_str = 'unknown'
            # The solutions status is unknown. The termination code
            # indicates why the optimizer terminated prematurely.
            print("The solution status is unknown.")
            symname, desc = mosek.Env.getcodedesc(mosek.rescode(int(M.getSolverIntInfo("optimizeResponse"))))
            print("   Termination code: {0} {1}".format(symname, desc))
        else:
            status_str = 'other'
            print("Another unexpected problem status {0} is obtained.".format(status))
    except OptimizeError as e:
        print("Optimization failed. Error: {0}".format(e))
    except SolutionError as e:
        status = M.getProblemStatus()
    except Exception as e:
        print("Unexpected error: {0}".format(e))

    if status_str in ['feasible', 'infeasible']:
        coeffs = {}
        for var in known_vars.keys():
            coeffs[var] = np.sum(ymat[np.where(positionsmatrix == var)])
        if feas_as_optim:
            # In feasibility-as-optimization problems, the certificate is offset
            # by the optimal value
            coeffs[1] += -primal
        vars_of_interest = {'sol': primal, 'G': xmat, 'Z': ymat,
                            'dual_certificate': coeffs, 'xi': xi_list}
        return vars_of_interest, primal, status_str
    else:
        return None, None, status_str

def solveSDP_MosekFUSION2(positionsmatrix: scipy.sparse.lil_matrix,
                         objective={1: 0.}, known_vars={0: 0., 1: 1.},
                         semiknown_vars={}, positive_vars=[],
                         verbose=0, feas_as_optim=False, solverparameters={},
                         var_lowerbounds={}, var_upperbounds={},
                         solve_dual=True) -> Tuple[Dict, float, str]:
    """Internal function to solve the SDP with the MOSEK Fusion API.

    Now follows an extended description of how the SDP is encoded. In general,
    it is prefered solve using the dual formulation, which is the default.      

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
        optimisation information such as the primal objective value, 
        the moment matrix, the dual values, the certificate and 
        a dictionary of values for the monomials, in the following keys in
        the same order: 'sol', 'G', 'Z', 'dual_certificate', 'xi'. 
        The second element is the objective value and the last is the
        problem status.
    """

    # TODO Make sure we have the correct sign when extracting constraint duals
    # TODO If we want to allow for subnormalised moment matrices, we need
    # to decouple the constant variable from the moment '1'. Currently in the
    # objective value, 1 is reserved for constants. However, this is not 
    # relevant for current applications.

    import mosek
    from mosek.fusion import Matrix, Model, ObjectiveSense, Expr, Domain, \
                             OptimizeError, SolutionError, \
                             AccSolutionStatus, ProblemStatus

    for x in positive_vars:
        try:
            if var_lowerbounds[x] < 0:
                var_lowerbounds[x] = 0
        except KeyError:
            var_lowerbounds[x] = 0
    
    try:
        del known_vars[0]  # TODO Fix a convention
    except KeyError:
        pass

    positionsmatrix = positionsmatrix.astype(np.uint16)
    mat_dim = positionsmatrix.shape[0]

    OBJ_CONSTANT_KEY = 1  # If we hash monomials differently, we might 
                          # need to change the hash of the constant/offset term
                          # from the number 1 to something else.
                          # The rest of the code should be insensitive to 
                          # the hash except this line (ideally).

    variables = set(positionsmatrix.flatten())

    F0 = scipy.sparse.lil_matrix((mat_dim,mat_dim))
    Fi = {}
    for x in variables:
        coeffmat = scipy.sparse.lil_matrix((mat_dim,mat_dim))
        coeffmat[scipy.sparse.find(positionsmatrix == x)[:2]] = 1
        Fi[x] = coeffmat

    # ! (From here onwards we no longer need positionsmatrix for anything.)

    # Remove variables that are fixed by known_vars from the list of
    # variables, and also remove the corresponding entries for its upper 
    # and lower bounds.
    for x, xval in known_vars.items():
        F0 += xval * Fi[x]
        variables.remove(x)
        # We do not delete Fi[x] because we need them later for the certificate.
        
        # Now update the bounds dictionaries with the known values (i.e.,
        # remove the entries for known variables). 
        if x in var_lowerbounds:
            lb = var_lowerbounds[x]
            if lb >= xval:
                # We warn the user when these are incompatible, but the
                # program will continue. 
                # TODO Should we remove this check? It is unlikely that
                # the bounds will be incompatible with fixed values, if the
                # user uses our program correctly.
                UserWarning(
                    "Lower bound {} for variable {}".format(lb, x) + 
                    " is incompatible with the known value {}.".format(xval) + 
                    " The lower bound will be ignored.")
            del var_lowerbounds[x]
        if x in var_upperbounds:
            ub = var_upperbounds[x]
            if ub <= xval:
                UserWarning(
                    "Upper bound {} for variable {}".format(ub, x) + 
                    " is incompatible with the known value {}.".format(xval) + 
                    " The upper bound will be ignored.")
            del var_upperbounds[x]


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
        # bounds of variables involved in LPI constraints? It is not clear to
        # me, but I am inclined to say no. For our usecase, it should not 
        # happen that we have one upper bound for one variable and lower bound
        # for the other variable such that the constraint can never be
        # satisfied.
        if x in var_lowerbounds and x2 in var_lowerbounds:
            del var_lowerbounds[x]
        if x in var_upperbounds and x2 in var_upperbounds:
            del var_upperbounds[x]

    # Before converting to MOSEK format, it is useful to keep indices of where
    # F0, Fi are nonzero, as it seems to be more difficult to extract later.
    ij_F0_nonzero = [(i,j) for (i,j) in zip(*F0.nonzero()) if j >= i]
    ij_Fi_nonzero = {}
    for x in variables:
        ij_Fi_nonzero[x] = [(i,j) for (i,j) in zip(*Fi[x].nonzero()) if j >= i]
    
    # Convert to MOSEK format.
    F0 = Matrix.sparse(*F0.shape, *F0.nonzero(), F0[F0.nonzero()].A[0])
    for x in variables:
        F = Fi[x]
        Fi[x] = Matrix.sparse(*F.shape, *F.nonzero(), F[F.nonzero()].A[0])

    # Find if the objective is constant.
    # We can safely assume that with the known_constraints, all known variables
    # in the objective have already been removed, and added to 
    # objective[OBJ_CONSTANT_KEY]. However, just to be certain, we will
    # do this step also here.
    for x, val in known_vars.items():
        if x in objective and x != 1:
            objective[OBJ_CONSTANT_KEY] += val * objective[x]
            del objective[x]
    constant_objective = False
    if list(objective.keys()) == [1]: 
        constant_objective = True
        if verbose > 1:
            print('Constant objective detected! Treating the problem as ' +
                  'a feasibility problem.')

    if var_lowerbounds:
        # The following dictionary, lowerbounded_var2idx, is because the L 
        # dual variable will only be defined for the variables with explicitly 
        # defined lower bounds (this is to avoid having to write a -infinity 
        # lower bound). as such, it is useful to have a dictionary that maps
        # a monomial x to the index of i of Li, where Li is the dual variable
        # of the constraint x - lb_x >= 0.
        lowerbounded_var2idx = {x: i for i, x in enumerate(var_lowerbounds)}
    if var_upperbounds:
        # Same as above.
        upperbounded_var2idx = {x: i for i, x in enumerate(var_upperbounds)}

    M = Model('InfSDP')

    if solve_dual:
        # Define variables
        Z = M.variable(Domain.inPSDCone(mat_dim))
        if var_lowerbounds:
            L = M.variable(len(var_lowerbounds), Domain.greaterThan(0))
        if var_upperbounds:
            U = M.variable(len(var_upperbounds), Domain.greaterThan(0))

        # Define objective function
        mosek_obj = Expr.dot(Z, F0)
        if var_lowerbounds:
            mosek_obj = Expr.sub(mosek_obj, 
                                 Expr.dot(L, list(var_lowerbounds.values())))
        if var_upperbounds:
            mosek_obj = Expr.add(mosek_obj,
                                 Expr.dot(U, list(var_upperbounds.values())))
        if not constant_objective:
            # If we are doing a relaxed feasibility test, then we want the
            # interpretation of the result being the maximum minimum eigenvalue.
            # However, if we add a constant to the objective, then this
            # interpretation breaks down, so we only add it if the objective
            # is not constant.
            mosek_obj = Expr.add(mosek_obj, float(objective[OBJ_CONSTANT_KEY]))
        
        # Set objective function
        M.objective(ObjectiveSense.Minimize, mosek_obj)

        # Add constraints
        ci_constraints = []
        for x in variables:
            F = Fi[x]
            lhs = Expr.dot(Z,F)
            ci = objective[x] if x in objective else 0
            lhs = Expr.add(lhs, float(ci))
            if x in var_upperbounds:
                lhs = Expr.sub(lhs, U.index(upperbounded_var2idx[x]))
            if x in var_lowerbounds:
                lhs = Expr.add(lhs, L.index(lowerbounded_var2idx[x]))

            # Tr Z Fi + ci - Ui + Li = 0
            ci_constraints.append(M.constraint(lhs, Domain.equalsTo(0)))
    
        if feas_as_optim:
            # Tr Z = 1
            ci_constraints.append(M.constraint(Expr.dot(Z, Matrix.eye(mat_dim)),
                                               Domain.equalsTo(1)))
    else:
        # Define variables
        x_mosek = M.variable(len(variables), Domain.unbounded())
        mon2xindex = {x: i for i, x in enumerate(variables)}

        # Add upper and lower bounds
        if var_lowerbounds:
            lb_constraints = []
            for x, val in var_lowerbounds.items():
                try:
                    # x_i - lb_i >= 0
                    lb_constraints.append(M.constraint(
                                        Expr.sub(x_mosek.index(mon2xindex[x]),
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
                                                 x_mosek.index(mon2xindex[x])),
                                        Domain.greaterThan(0)))
                except KeyError:
                    pass

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
                mosek_obj = float(objective[OBJ_CONSTANT_KEY])
        else:
            mosek_obj = float(objective[OBJ_CONSTANT_KEY])
            for xi, ci in objective.items():
                if xi != OBJ_CONSTANT_KEY:
                    mosek_obj = Expr.add(mosek_obj,
                                        Expr.mul(float(ci),
                                                x_mosek.index(mon2xindex[xi])))
        
        M.objective(ObjectiveSense.Maximize, mosek_obj)

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

            # If feasible, make sure the bounds are satisfied!
            # TODO Maybe remove this check in the future if we find this
            # error never appears
            for x, lb in var_lowerbounds.items():
                assert x_values[x] - lb >= 0, "Lower bound violated!"
            for x, ub in var_upperbounds.items():
                assert ub - x_values[x] >= 0, "Upper bound violated!"
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
    except SolutionError as e:
        status = M.getProblemStatus()
    except Exception as e:
        print("Unexpected error: {0}".format(e))

    if status_str in ['feasible', 'infeasible']:
        certificate = {}
        for x in known_vars:
            support = Fi[x].nonzero()
            certificate[x] = np.dot(ymat[support], Fi[x][support].A[0])
        if var_lowerbounds:
            Lvalues = L.level() if solve_dual else [-c.dual() for c in lb_constraints]
            for x, val in var_lowerbounds.items():
                certificate[1] -= Lvalues[lowerbounded_var2idx[x]] * val
        if var_upperbounds:
            Uvalues = U.level() if solve_dual else [-c.dual() for c in ub_constraints]
            for x, val in var_upperbounds.items():
                certificate[1] += Uvalues[upperbounded_var2idx[x]] * val

        vars_of_interest = {'sol': primal, 'G': xmat, 'Z': ymat,
                            'dual_certificate': certificate,
                            'xi': x_values}

        return vars_of_interest, primal, status_str
    else:
        return None, None, status_str
