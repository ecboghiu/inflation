import gc
import os
import sys
from typing import Tuple

import cvxpy as cp
import mosek
import numpy as np
import picos
import scipy.sparse
from mosek.fusion import *  # Todo make so you don't import this if you dont use the mosek solver to reduce dependencies
from scipy.io import loadmat
from tqdm import tqdm

def load_MATLAB_SDP(filename_SDP_info: str, use_semiknown: bool = False):
    SDP_data = loadmat(filename_SDP_info)
    positionsmatrix = SDP_data['G'] - 1  # Offset index to Python convention (== counting from 0, not 1)
    known_vars_array = SDP_data['known_moments'][0]

    if use_semiknown:
        semiknown_vars_array = SDP_data['propto']  # Index in MATLAB convention!
        semiknown_vars_array[:, [0,2]] += -1  # Offset the first and second column index to Python convention
    else:
        semiknown_vars_array = []  # Needed for del to work. Would be happy not to have to write this

    return positionsmatrix, [], known_vars_array, semiknown_vars_array

def solveSDP_MosekFUSION(   positionsmatrix: scipy.sparse.lil_matrix, objective: dict = {},
                            known_vars=[0, 1], semiknown_vars=[], positive_vars=[],
                            verbose: int = 0,
                            pure_feasibility_problem: bool = False,
                            solverparameters: dict = {}):

    semiknown_vars = np.array(semiknown_vars)
    positive_vars = np.array(positive_vars)
    
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

    M = Model('InfSDP')
    if solve_dual:
        Z = M.variable("Z", Domain.inPSDCone(mat_dim))
        if objective:
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
            if objective:
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
        if primal >= INFEASIBILITY_THRESHOLD:
            coeffs = coeffs*0  # Certificates are meaningless here
        if primal < INFEASIBILITY_THRESHOLD:
            coeffs[1] += -primal  # If the minimum eiganvalue is negative, the certificate is that this minimum eigenvalue is  
    vars_of_interest = {'sol': primal, 'G': xmat, 'dual_certificate': coeffs, 'Z': ymat, 'xi': xi_list}
    return vars_of_interest, primal


def solveSDP_CVXPY(positionsmatrix, objective,
                    known_vars=[0, 1], semiknown_vars=[], positive_vars=[],
                    pure_feasibility_problem: bool = False,
                    verbose: int = 0,
                    solverparameters: dict = {}):
    if pure_feasibility_problem:
        if verbose > 0:
            Warning("Cannot get infeasibility certificates in CVXPY. Suggestion for "+
                    "certificate extraction: set 'interpreter' flag to MOSEKFusion."+
                    "Proceeding to solve the problem in CVXPY.")
        pure_feasibility_problem = False

    '''
    if 'solver' in solverparameters:
        solver = solverparameters['solver']
        if solver == 'mosek':
            solver = cp.MOSEK
        elif solver == 'cvxopt':
            solver = cp.CVXOPT
        elif solver == 'scs':
            solver = cp.SCS
        elif solver == 'gurobi':
            solver = cp.GUROBI
        else:
            if type(solver) == str:
                Warning('Solver not recognized. Using CVXOPT instead.')
            # Now 'solver' containts what the user specified in solverparameters
    else:
        solver = cp.CVXOPT
    '''

    # The +1 is because positionsmatrix starts counting from 0,
    # but then we have two reserved variables for 0 and 1
    if np.array(semiknown_vars).size > 0:
        # TODO find a more elegant way to do this?
        # When using proportionality constraints, we might get an entry is 
        # proportional to a variable that is not found at any other entry, thus
        # the total number of variables is not just the number of unknown entries
        # of the moment matrix.

        # However, if no new variables are to be found, the index of the biggest 
        # variable in semiknown_vars_array will be smaller than the maximum in all
        # the entries of positionsmatrix.
        nr_variables = max([int(np.max(semiknown_vars)), int(np.max(positionsmatrix))]) + 1
    else:
        nr_variables = np.max(positionsmatrix) + 1
    nr_known = len(known_vars)
    nr_unknown = nr_variables - nr_known

    F0 = scipy.sparse.lil_matrix(positionsmatrix.shape)
    for variable in range(nr_known):
        F0[np.where(positionsmatrix == variable)] = known_vars[variable]

    Fi = []
    for var in range(nr_known, nr_variables):
        F = scipy.sparse.lil_matrix(positionsmatrix.shape)
        F[np.where(positionsmatrix == var)] = 1
        Fi.append(F)

    x = cp.Variable(nr_unknown, name='x')

    G = F0
    for i, F in enumerate(Fi):
        G = G + x[i] * F

    if not objective:
        Id = cp.diag(np.ones(positionsmatrix.shape[0]))
        if pure_feasibility_problem:
            obj = 0
            constraints = [ G >> 0 ]
        else:
            obj = cp.Variable()
            constraints = [ G - obj*Id >> 0 ]
    else:
        coeffs = np.zeros(nr_unknown)
        offset = 0
        for variable, coefficient in objective.items():
            if variable > 1:
                # The 2 is correcting for the constants 0 and 1 being indexed
                # as variables
                coeffs[variable - 2] = coefficient
            else:
                offset = coefficient
        obj = coeffs @ x + offset#*G[0,0]
        constraints = [ G >> 0 ]

    if np.array(positive_vars).size > 0:
        for var in positive_vars:
            constraints += [ x[var-nr_known] >= 0 ]

    if np.array(semiknown_vars).size > 0:
        for var1, const, var2 in semiknown_vars:
            constraints += [ x[var1-nr_known] == const * x[var2-nr_known] ]

    constraints += [x >= -1]
    constraints += [x <= +1]

    prob = cp.Problem(cp.Maximize(obj), constraints)

    # TODO: Is it actually a good idea to use del and garbage collector? It might be useless...
    #del positionsmatrix, known_vars, semiknown_vars, Id, F0, Fi[:], Fi
    #gc.collect()

    if not solverparameters:
        prob.solve(verbose=bool(verbose))
    else:
        prob.solve(**solverparameters)

    # TODO: Get dual certificate in symbolic form
    coeffs = np.zeros(nr_known, dtype=np.float64)
    vars_of_interest = {'sol': prob.solution, 'G': G.value, 'dual_certificate': coeffs}
    if not objective:
        # If it is meaningful to get a dual certificate, we do so.
        if not np.array(semiknown_vars).size > 0:
            Z = constraints[0].dual_value
            coeffs = np.zeros(nr_known, dtype=np.float64)
            for var in range(nr_known):
                coeffs[var] = np.sum(Z[np.where(positionsmatrix == var)])
            coeffs = -coeffs  # s.t. we have TrZF0 >= 0 instead of -TrZF0 <= 0

            vars_of_interest = {'sol': prob.solution, 'G': G.value, 'dual_certificate': coeffs}

    return vars_of_interest, obj.value




def solveSDP_PICOS(positionsmatrix, objective, known_vars=[0, 1], semiknown_vars=[], positive_vars=[],
             solver = 'cvxopt', verbose: int = 0,
             pure_feasibility_problem: bool = False):
    """
    Solves an SDP and returns the solution object and the optimum objective. Takes as input the filename for a .mat file
    with the different variables that are needed.
    """
    # The +1 is because positionsmatrix starts counting from 0,
    # but then we have two reserved variables for 0 and 1
    if np.array(semiknown_vars).size > 0:
        # TODO find a more elegant way to do this?
        # When using proportionality constraints, we might get an entry is 
        # proportional to a variable that is not found at any other entry, thus
        # the total number of variables is not just the number of unknown entries
        # of the moment matrix.

        # However, if no new variables are to be found, the index of the biggest 
        # variable in semiknown_vars_array will be smaller than the maximum in all
        # the entries of positionsmatrix.
        nr_variables = max([int(np.max(semiknown_vars)), int(np.max(positionsmatrix))]) + 1
    else:
        nr_variables = np.max(positionsmatrix) + 1
    nr_known = len(known_vars)
    nr_unknown = nr_variables - nr_known

    F0 = scipy.sparse.lil_matrix(positionsmatrix.shape)
    for variable in range(nr_known):
        F0[np.where(positionsmatrix == variable)] = known_vars[variable]

    Fi = []
    for var in range(nr_known, nr_variables):
        F = scipy.sparse.lil_matrix(positionsmatrix.shape)
        F[np.where(positionsmatrix == var)] = 1
        Fi.append(F)

    P = picos.Problem()

    #x = cp.Variable(nr_unknown)
    x = picos.RealVariable("x", nr_unknown)
    
    G = F0
    for i in range(len(Fi)):
        G = G + x[i] * Fi[i]

    if not objective:
        Id = cp.diag(np.ones(positionsmatrix.shape[0]))  # do sparse diag
        if pure_feasibility_problem:
            obj = 0
            P.add_constraint( G  >> 0 )
        else:
            #obj = cp.Variable()
            lam = picos.RealVariable("lam")
            obj = lam
            P.add_constraint( G - lam*Id >> 0 )

    else:
        Id = 0
        coeffs = np.zeros(nr_unknown)
        for variable, coefficient in objective.items():
            # The 2 is correcting for the constants 0 and 1 being indexed
            # as variables
            coeffs[variable - 2] = coefficient
        obj = coeffs @ x
        constraints = [ G >> 0 ]

    if np.array(positive_vars).size > 0:
        for var in positive_vars:
            P.add_constraint( x[var-nr_known] >= 0 )

    if np.array(semiknown_vars).size > 0:
        for var1, const, var2 in semiknown_vars:
            P.add_constraint( x[var1-nr_known] == const * x[var2-nr_known] )

    P.set_objective("max", obj)


    P.options['dualize']=True

    #prob.solve(solver=solver, verbose=verbose)
    #prob.solve(solver=cp.CVXOPT, verbose=verbose)
    #prob.solve(solver=cp.SCS)
    sol = P.solve(solver='mosek', verbose=True)

    # TODO: Get dual certificate in symbolic form

    if not objective:
        if not np.array(semiknown_vars).size > 0:
            Z = constraints[0].dual_value
            coeffs = np.zeros(nr_known, dtype=np.float64)
            for var in range(nr_known):
                coeffs[var] = np.sum(Z[np.where(positionsmatrix == var)])
            coeffs = -coeffs  # s.t. we have TrZF0 >= 0 instead of -TrZF0 <= 0
            vars_of_interest = {'sol': P.solution, 'G': G.value, 'dual_certificate': coeffs}
    else:
        vars_of_interest = {'sol': P.solution, 'G': G.value}
    return vars_of_interest, obj.value



def read_from_sdpa(filename, verbose=0):
    with open(filename, 'r') as file:
        problem = file.read()
    _, nvars, nblocs, blocstructure, obj = problem.split('\n')[:5]

    if verbose > 0:
        iterator = tqdm(problem.split('\n')[5:-1], desc='Reading problem', disable=verbose)
    else:
        iterator = problem.split('\n')[5:-1]
    mat = [list(map(float, row.split('\t'))) for row in iterator]
    return mat, obj


def create_array_from_sdpa(sdpa_mat, verbose=0):
    sdpa_mat = np.array(sdpa_mat)
    size = int(max(sdpa_mat[:, 3]))
    mat = np.zeros((size, size, 2))
    if verbose > 0:
        iterator = tqdm(sdpa_mat, disable=verbose)
    else:
        iterator = sdpa_mat
    for var, _, i, j, val in iterator:
        mat[int(i - 1), int(j - 1)] = np.array([var, val])
        mat[int(j - 1), int(i - 1)] = np.array([var, val])
    return mat

def extract_from_ncpol(problem, verbose=0):
    '''
    Extracts the moment matrix, monomial list and objective function from an
    SdpRelaxation object from ncpol2sdpa
    '''
    prob_filename = 'temp_SDPAProblem.dat-s'
    mon_filename  = 'temp_monomials.txt'
    problem.write_to_file(prob_filename)
    problem.save_monomial_index(mon_filename)
    momentmatrix, objective, monomials = read_problem_from_file(prob_filename,
                                                                mon_filename,
                                                                verbose)
    os.remove(prob_filename)
    os.remove(mon_filename)

    return momentmatrix, objective, monomials

def read_problem_from_file(filename_momentmatrix,
                           filename_monomials,
                           verbose=0):
    problem, obj = read_from_sdpa(filename_momentmatrix, verbose)
    problem_arr  = create_array_from_sdpa(problem, verbose)
    # Reserve variables 0 and 1 for constant entries with 0 and 1, respectively
    # The original notation puts constants in the variable 0, so we must split
    # this and have proper variables start counting from 2
    # Have the variables begin from 2
    problem_arr[:,:,0] += 1
    # The constants, which are now under variable 1, are split in zeros and ones
    problem_arr[(problem_arr[:,:,0] == 1) & (problem_arr[:,:,1] == 0)] = [0, 1]
    problem_arr[(problem_arr[:,:,0] == 1) & (problem_arr[:,:,1] == 1)] = [1, 1]
    obj = np.array(eval(obj.replace('{', '[').replace('}', ']')))
    nonzerovariables = np.nonzero(obj)[0]
    nonzerocoefficients = obj[nonzerovariables]
    # Indexing of variables begins in 1 in ncpol2sdpa, plus the extra index from
    # splitting the constants makes an offset of 2
    nonzerovariables += 2
    obj = dict(zip(nonzerovariables, nonzerocoefficients))
    monomials_list = np.genfromtxt(filename_monomials,
                                   dtype=str, skip_header=1).astype(object)
    # Monomials in ncpol2sdpa begin from 1, but now they must begin from 2
    for idx in range(len(monomials_list)):
        monomials_list[idx][0] = str(int(monomials_list[idx][0]) + 1)
    return problem_arr, obj, monomials_list
