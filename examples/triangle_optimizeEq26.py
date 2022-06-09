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
cws = cws + os.sep + os.pardir  # Go one folder above UnifiedInflation/examples -> UnifiedInflation/
sys.path.append(cws)
###############################################################################

import itertools

import numpy as np
from causalinflation.InflationProblem import InflationProblem
from causalinflation.InflationSDP import InflationSDP
from causalinflation.useful_distributions import P_GHZ, P_W_array
import sympy as sp

from causalinflation.general_tools import to_numbers
from causalinflation.fast_npa import calculate_momentmatrix, to_name

if __name__ == '__main__':  # Necessary for parallel computation, used in ncpol2sdpa
    outcomes_per_party = [2, 2, 2]
    settings_per_party = [1, 1, 1]
    dag = {"h1": ["v1", "v2"],
           "h2": ["v1", "v3"],
           "h3": ["v2", "v3"]}
    InfProb = InflationProblem( dag=dag,
                                outcomes_per_party=outcomes_per_party,
                                settings_per_party=settings_per_party,
                                inflation_level_per_source=[2, 2, 2],
                                names=['A', 'B', 'C'] )

    InfSDP = InflationSDP(InfProb, commuting=False, verbose = 2)

    # InfSDP.build_columns('physical222', max_monomial_length=2)
    # cols_S2 = InfSDP.generating_monomials
    # InfSDP.build_columns('local1')
    # cols_loc1 = InfSDP.generating_monomials

    # cols_S2_name = [to_name(np.array(m), InfProb.names) for m in cols_S2[1:]]
    # cols_loc1_name = [to_name(np.array(m), InfProb.names) for m in cols_loc1[1:]]
    # cols = cols_S2_name
    # for m in cols_loc1_name:
    #     if m not in cols_S2_name:
    #         cols.append(m)
    # #cols = cols_S2_name.union(cols_loc1_name)
    # cols_num = [[0]]
    # for m in cols:
    #     cols_num.append(to_numbers(m, InfProb.names))

    filename_label = 'eq26'
    col_specification = [[], [0], [2], [0, 0], [0, 1], [1, 1], [0, 2], [2, 2], [1, 2], [0, 1, 2]]
    InfSDP.generate_relaxation( column_specification='local1',#col_specification,#cols_num,#'physical', # col_specification,
                                #max_monomial_length = 4,
                                filename_label=filename_label,
                                find_physical_monomials=True,
                                sandwich_positivity=True,
                                parallel=False,
                                load_from_file=False,
                                use_numba=True)


    p_W = P_W_array(visibility=1.0)

    measurements = InfSDP.measurements  # This reads as measurements[party][inflation_source][setting][outcome]. TODO remove "inflation_source" as the user doesnt care

    sum_abc_pW_squared = np.multiply(p_W,p_W).sum()

    sum_abc_p_obs_times_p_W = 0
    sum_abc_p_obs_squared = 0

    for a, b, c in itertools.product(*[range(o) for o in outcomes_per_party]):
        # < A_11 * A_22 * B_11 * B_22 * C_11 * C_22 >
        p_obs_squared = 1
        p_obs_squared *= measurements[0][0][0][a] * measurements[0][3][0][a] if a==0 else (1-measurements[0][0][0][0]) * (1-measurements[0][3][0][0])
        p_obs_squared *= measurements[1][0][0][b] * measurements[1][3][0][b] if b==0 else (1-measurements[1][0][0][0]) * (1-measurements[1][3][0][0])
        p_obs_squared *= measurements[2][0][0][c] * measurements[2][3][0][c] if c==0 else (1-measurements[2][0][0][0]) * (1-measurements[2][3][0][0])

        sum_abc_p_obs_squared += p_obs_squared

        # p_W(a,b,c) * < A_11 * B_11 * C_11 >
        p_obs_times_p_W = p_W[a, b, c, 0, 0, 0]
        p_obs_times_p_W *= measurements[0][0][0][0] if a==0 else (1-measurements[0][0][0][0])
        p_obs_times_p_W *= measurements[1][0][0][0] if b==0 else (1-measurements[1][0][0][0])
        p_obs_times_p_W *= measurements[2][0][0][0] if c==0 else (1-measurements[2][0][0][0])

        sum_abc_p_obs_times_p_W += p_obs_times_p_W


    objective = sum_abc_pW_squared + sum_abc_p_obs_squared - 2 * sum_abc_p_obs_times_p_W
    objective = sp.expand(objective)

    InfSDP.set_objective(objective=objective, direction='min')  # By default it maximizes
    InfSDP.write_to_file('inflationMATLAB_'+filename_label+'.mat')

    InfSDP.solve(interpreter="MOSEKFusion")

    print(InfSDP.primal_objective)
