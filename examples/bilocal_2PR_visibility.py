import os
import sys
# Locate the script in UnifiedInflation/examples and add to path one folder before, that is, UnifiedInflation/
# in order to be able to import quantuminflation
# ! Note: I found online that "__file__" sometimes can be problematic, So I'm using the solution provided in
# https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python?lq=1
from inspect import getsourcefile
from os.path import abspath

cws = abspath(getsourcefile(lambda:0))
cws = os.sep.join(cws.split(os.sep)[:-1])  # Remove the script filename to get the directory of the script
cws = cws + os.sep + os.pardir             # Go one folder above UnifiedInflation/examples -> UnifiedInflation/
sys.path.append(cws)

################################################################################

import numpy as np
from causalinflation.InflationProblem import InflationProblem
from causalinflation.InflationSDP import InflationSDP
from causalinflation.useful_distributions import (P_2PR_array, P_CHSH_array,
                                                   P_GHZ_array, P_Mermin_array,
                                                   P_W_array)

InfProb = InflationProblem( dag={"h1": ["v1", "v2"],
                                 "h2": ["v2", "v3"]},
                            outcomes_per_party=[2, 2, 2],
                            settings_per_party=[2, 2, 2],
                            inflation_level_per_source=[2, 2],
                            names=['A', 'B', 'C'])

InfSDP = InflationSDP(InfProb, commuting=True, verbose=2)

# InfSDP.build_columns('physical222', max_monomial_length=2)
# cols_S2 = InfSDP.generating_monomials
# InfSDP.build_columns('physical111')
# cols_loc1 = InfSDP.generating_monomials

# cols_S2_name = set([to_name(np.array(m), InfProb.names) for m in cols_S2[1:]])
# cols_loc1_name = set([to_name(np.array(m), InfProb.names) for m in cols_loc1[1:]])
# cols = cols_S2_name.union(cols_loc1_name)
# cols_num = [[0]]
# for m in cols:
#     cols_num.append(to_numbers(m, InfProb.names))

InfSDP.generate_relaxation(column_specification='npa3', max_monomial_length=2)

# Do a bisection
InfSDP.verbose = 0

tol_vis = 1e-4
v0, v1 = 0, 1
vm = (v0 + v1)/2

iteration = 0
while abs(v1 - v0) >= tol_vis and iteration < 20:
    p = P_2PR_array(visibility=vm)
    InfSDP.set_distribution(p, use_lpi_constraints=True)
    InfSDP.solve()

    print("vm =", iteration, vm, "\tmax min eigenvalue:", InfSDP.primal_objective)
    iteration += 1
    if InfSDP.primal_objective > 0:
        v0 = vm
        vm = (v0 + v1)/2
    elif InfSDP.primal_objective < 0:
        v1 = vm
        vm = (v0 + v1)/2
    elif abs(InfSDP.primal_objective) <= 1e-7:
        break
    else:
        raise "Something happened..."
    
