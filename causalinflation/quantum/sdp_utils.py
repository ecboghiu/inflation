import numpy as np
import scipy.sparse
import sys

def solveSDP_MosekFUSION(positionsmatrix: scipy.sparse.lil_matrix,
                         objective: dict = {}, known_vars=[0, 1],
                         semiknown_vars=[], positive_vars=[], verbose: int = 0,
                         pure_feasibility_problem: bool = False,
                         solverparameters: dict = {}):

    import mosek
    from mosek.fusion import Matrix, Model, ObjectiveSense, Expr, Domain, \
                             OptimizeError, SolutionError, \
                             AccSolutionStatus, ProblemStatus

    semiknown_vars = np.array(semiknown_vars)
    positive_vars  = np.array(positive_vars)

    solve_dual = True
    if solverparameters:
        solve_dual = True if 'solve_dual' not in solverparameters else solverparameters['solve_dual']

    use_positive_vars = False
    if positive_vars.size > 0:
        use_positive_vars = True

    # The +1 is because positionsmatrix starts counting from 0,
    # but then we have two reserved variables for 0 and 1
    positionsmatrix = positionsmatrix.astype(np.uint16)
    if semiknown_vars.size > 0:
        # TODO find a more elegant way to do this?
        # When using proportionality constraints, we might get an entry is
        # proportional to a variable that is not found at any other entry, thus
        # the total number of variables is not just the number of unknown entries
        # of the moment matrix.
        # However, if no new variables are to be found, the index of the biggest
        # variable in semiknown_vars_array will be smaller than the maximum in all
        # the entries of positionsmatrix.
        nr_variables = max([int(np.max(semiknown_vars)), int(positionsmatrix.max())]) + 1
    else:
        nr_variables = np.max(positionsmatrix) + 1
    nr_known = len(known_vars)
    nr_unknown = nr_variables - nr_known

    F0 = scipy.sparse.lil_matrix(positionsmatrix.shape)
    for variable in range(nr_known):
        F0[scipy.sparse.find(positionsmatrix == variable)[:2]] = known_vars[variable]
    F0 = Matrix.sparse(*F0.shape, *F0.nonzero(), F0[F0.nonzero()].todense().A[0])

    # List of empty sparse matrices
    Fi = []
    for variable in range(nr_known, nr_variables):
        F = scipy.sparse.lil_matrix(positionsmatrix.shape)
        F[scipy.sparse.find(positionsmatrix == variable)[:2]] = 1  # Set to 1 where the unknown variable is
        #print(F.todense())
        #F = Matrix.sparse(*F.shape, *F.nonzero(), F[F.nonzero()].todense().A[0])
        Fi.append(F)

    if semiknown_vars.size > 0:
        Fii = []
        x1 = semiknown_vars[:, 0].astype(int)
        k  = semiknown_vars[:, 1]
        x2 = semiknown_vars[:, 2].astype(int)
        for idx, variable in enumerate(range(nr_known, nr_variables)):
            if variable in x1:
                Fi[x2[idx]-nr_known] = Fi[x2[idx]-nr_known] + k[idx] * Fi[x1[idx]-nr_known]
            else:
                Fii.append(Fi[idx])
        Fi = Fii

        nr_variables = nr_variables - semiknown_vars.shape[0]

    # Convert to MOSEK format
    for i in range(len(Fi)):
        F = scipy.sparse.lil_matrix(Fi[i])
        Fi[i] = Matrix.sparse(*F.shape, *F.nonzero(), F[F.nonzero()].todense().A[0]) #Matrix.sparse(*F.shape, *F.nonzero(), F[F.nonzero()].todense().A[0])

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
            M.objective(ObjectiveSense.Minimize, Expr.add(float(objective[1]), Expr.dot(Z, F0)))
            for i, F in enumerate(Fi):
                vars_in_obj = np.array(sorted(list(objective.keys()))[1:], dtype=int)-2
                if i in vars_in_obj:
                    if use_positive_vars and i in positive_vars:
                            ci_constraints.append(M.constraint(Expr.dot(Z,F), Domain.lessThan(-float(objective[i+2]))))  # TODO check that variables which are known dont appear in objective
                    else:
                        ci_constraints.append(M.constraint(Expr.dot(Z,F), Domain.equalsTo(-float(objective[i+2]))))
                else:
                    ci_constraints.append(M.constraint(Expr.dot(Z,F), Domain.equalsTo(0)))
        else:
            M.objective(ObjectiveSense.Minimize, Expr.dot(Z, F0))
            ci_constraints = []
            for i, F in enumerate(Fi):
                F = Fi[i]
                if use_positive_vars and i in positive_vars:
                    ci_constraints.append(M.constraint(Expr.dot(Z,F), Domain.lessThan(0)))
                else:
                    ci_constraints.append(M.constraint(Expr.dot(Z,F), Domain.equalsTo(0)))
            if not pure_feasibility_problem:
                ci_constraints.append(M.constraint(Expr.dot(Z,Matrix.eye(mat_dim)), Domain.equalsTo(1)))
    else:
        # ! The primal formulation uses a lot more RAM and is slower!! Only use if the problem is not too big
        x = M.variable(nr_variables, Domain.unbounded())
        for var in range(nr_known):
            M.constraint(x.index(var), Domain.equalsTo(float(known_vars[var])))

        if use_positive_vars:
            for i in positive_vars:
                M.constraint(x.index(i), Domain.greaterThan(0))

        #G = x.pick(positionsmatrix.astype(int).reshape((1,int(positionsmatrix.size)))).reshape([mat_dim, mat_dim])
        #G = Expr.reshape(x.pick(positionsmatrix.astype(int).flatten()),[mat_dim, mat_dim])
        #G = M.variable("G", Domain.unbounded([mat_dim, mat_dim]))
        #const = M.constraint(G, Domain.inPSDCone(mat_dim))
        G = M.variable("G", Domain.inPSDCone(mat_dim))
        if pure_feasibility_problem:
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
            # The solutions status is unknown. The termination code
            # indicates why the optimizer terminated prematurely.
            print("The solution status is unknown.")
            symname, desc = mosek.Env.getcodedesc(mosek.rescode(int(M.getSolverIntInfo("optimizeResponse"))))
            print("   Termination code: {0} {1}".format(symname, desc))
        else:
            print("Another unexpected problem status {0} is obtained.".format(status))
    except OptimizeError as e:
        print("Optimization failed. Error: {0}".format(e))
    except SolutionError as e:
        status = M.getProblemStatus()
    except Exception as e:
        print("Unexpected error: {0}".format(e))

    INFEASIBILITY_THRESHOLD = -1e-6

    vars_of_interest = {}
    coeffs = np.zeros(nr_known, dtype=np.float64)
    for var in range(nr_known):
        coeffs[var] = np.sum(ymat[np.where(positionsmatrix == var)])
    if not pure_feasibility_problem and not objective:  # i.e., if doing a relaxed feasibility problem, a maximization of the minimum eigenvalue
        coeffs[1] += -primal  # If the minimum eiganvalue is negative, the certificate is that this minimum eigenvalue is
    vars_of_interest = {'sol': primal, 'G': xmat, 'dual_certificate': coeffs, 'Z': ymat, 'xi': xi_list}
    return vars_of_interest, primal
