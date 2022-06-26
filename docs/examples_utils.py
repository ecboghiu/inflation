from copy import deepcopy
import itertools
import numpy as np

def feasibility_bisection(InfSDP, distribution, tol_vis=1e-4, verbose=0, max_iter=20):
    v0, v1 = 0, 1
    vm = (v0 + v1)/2
    
    InfSDP.verbose = 0
    iteration = 0
    last_good = deepcopy(InfSDP)

    while abs(v1 - v0) >= tol_vis and iteration < max_iter:
        p = distribution(visibility=vm)
        InfSDP.set_distribution(p, use_lpi_constraints=True)
        InfSDP.solve()
        if verbose:
            print("max(min eigval):", "{:10.4g}".format(
                InfSDP.primal_objective), "\tvisibility =", iteration, "{:.4g}".format(vm))
        iteration += 1
        if InfSDP.primal_objective >= 0:
            v0 = vm
            vm = (v0 + v1)/2
        elif InfSDP.primal_objective < 0:
            v1 = vm
            vm = (v0 + v1)/2
            last_good = deepcopy(InfSDP)
        if abs(InfSDP.primal_objective) <= 1e-7:
            break
    
    return last_good, vm

def probfunc2array(func, noise, outcomes_per_party, settings_per_party):
    p = np.zeros([*outcomes_per_party, *settings_per_party])
    for a,b,c,x,y,z in itertools.product(*[range(i) for i in [*outcomes_per_party, *settings_per_party]]):
        p[a,b,c,x,y,z] = func(a,b,c,x,y,z,noise)
    return p


