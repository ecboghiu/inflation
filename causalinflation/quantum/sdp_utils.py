import numpy as np
import scipy.sparse
import sys

def solveSDP_MosekFUSION(positionsmatrix: scipy.sparse.lil_matrix,
                         objective={1: 0.}, known_vars={0: 0., 1: 1.},
                         semiknown_vars={}, positive_vars=[],
                         verbose=0, feas_as_optim=False, solverparameters={}):

    import mosek
    from mosek.fusion import Matrix, Model, ObjectiveSense, Expr, Domain, \
                             OptimizeError, SolutionError, \
                             AccSolutionStatus, ProblemStatus

    positive_vars  = np.array(positive_vars)

    solve_dual = True
    if solverparameters:
        solve_dual = True if 'solve_dual' not in solverparameters else solverparameters['solve_dual']

    use_positive_vars = False
    if positive_vars.size > 0:
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

    constant_objective = False
    if list(objective.keys()) == [1]: # If there are only constants in the objective, treat it as a feasibility problem!
        constant_objective = True
        if verbose > 1:
            print('Constant objective detected! Treating the problem as ' +
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
