import os
import sys
# Locate the script in UnifiedInflation/examples and add to path one folder before, that is, UnifiedInflation/
# in order to be able to import quantuminflation
# ! Note: I found online that "__file__" sometimes can be problematic, So I'm using the solution provided in
# https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python?lq=1
from inspect import getsourcefile
from os.path import abspath
from tabnanny import verbose

from sympy import hyper

cws = abspath(getsourcefile(lambda:0))
cws = os.sep.join(cws.split(os.sep)[:-1])  # Remove the script filename to get the directory of the script
cws = cws + os.sep + os.pardir  # Go one folder above UnifiedInflation/examples -> UnifiedInflation/
sys.path.append(cws)
###############################################################################

import itertools

import numpy as np
import scipy
import sympy as sp
from causalinflation.fast_npa import calculate_momentmatrix
from causalinflation.InflationProblem import InflationProblem
from causalinflation.InflationSDP import InflationSDP
from causalinflation.useful_distributions import (P_CHSH_array, P_GHZ_array,
                                                   P_Mermin_array, P_W_array)


# Standard Alice and Bob scenario
InfProb = InflationProblem( dag= {"h1": ["v1", "v2"]},
                        outcomes_per_party=[2, 2],
                        settings_per_party=[2, 2],
                        inflation_level_per_source=[1],
                        names=['A', 'B'])

InfSDP = InflationSDP(InfProb, commuting=False, verbose=2)

InfSDP.generate_relaxation(column_specification='npa1')

meas = InfSDP.measurements
A0, A1, B0, B1 = 1-2*meas[0][0][0][0], 1-2*meas[0][0][1][0], 1-2*meas[1][0][0][0], 1-2*meas[1][0][1][0]
obj =  sp.expand(A0*B0 + A0*B1 + A1*B0 - A1*B1)
print(obj)
InfSDP.set_objective(objective=obj, direction='max')
InfSDP.solve(interpreter='MOSEKFusion')  # By default it maximizes
print("maximum of CHSH:", InfSDP.primal_objective)
