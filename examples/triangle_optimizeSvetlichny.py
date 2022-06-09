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
from causalinflation.useful_distributions import P_GHZ
import sympy as sp

from causalinflation.general_tools import to_numbers
from causalinflation.fast_npa import calculate_momentmatrix, to_name


InfProb = InflationProblem( dag={"h1": ["v1", "v2"],
                                 "h2": ["v1", "v3"],
                                 "h3": ["v2", "v3"]},
                            outcomes_per_party=[2, 2, 2],
                            settings_per_party=[2, 2, 2],
                            inflation_level_per_source=[2, 2, 2],
                            names=['A', 'B', 'C'] )

InfSDP = InflationSDP(InfProb, commuting=False, verbose = 2)

InfSDP.build_columns('physical222', max_monomial_length=2)
cols_S2 = InfSDP.generating_monomials
InfSDP.build_columns('local1')
cols_loc1 = InfSDP.generating_monomials

cols_S2_name = [to_name(np.array(m), InfProb.names) for m in cols_S2[1:]]
cols_loc1_name = [to_name(np.array(m), InfProb.names) for m in cols_loc1[1:]]
cols = cols_S2_name
for m in cols_loc1_name:
    if m not in cols_S2_name:
        cols.append(m)
#cols = cols_S2_name.union(cols_loc1_name)
cols_num = [[0]]
for m in cols:
    cols_num.append(to_numbers(m, InfProb.names))

col_specification = [[], [0], [1], [2], [0, 0], [0, 1], [1, 1], [0, 2], [2, 2], [1, 2]]
InfSDP.generate_relaxation(column_specification=col_specification)

measurements = InfSDP.measurements  # This reads as measurements[party][inflation_source][setting][outcome]. TODO remove "inflation_source" as the user doesnt care
A0 = sp.S.One - 2*measurements[0][0][0][0]
A1 = sp.S.One - 2*measurements[0][0][1][0]
B0 = sp.S.One - 2*measurements[1][0][0][0]
B1 = sp.S.One - 2*measurements[1][0][1][0]
C0 = sp.S.One - 2*measurements[2][0][0][0]
C1 = sp.S.One - 2*measurements[2][0][1][0]
objective = sp.expand(A1*B0*C0 +
                        A0*B1*C0 +
                        A0*B0*C1 +
                        -A1*B1*C1 +
                        -A0*B1*C1 +
                        -A1*B0*C1 +
                        -A1*B1*C0 +
                        A0*B0*C0 )

InfSDP.set_objective(objective=objective)  # By default it maximizes
InfSDP.write_to_file('inflationMATLAB.mat')

InfSDP.solve(interpreter="MOSEKFusion")

print(InfSDP.primal_objective)
