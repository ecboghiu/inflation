import itertools

import numpy as np
from causalinflation import InflationProblem, InflationSDP

outcomes_per_party = [2, 4, 2, 2]
settings_per_party = [2, 1, 2, 1]
dag = {"h1": ["A", "B", "E"],
       "h2": ["B", "C", "E"]}
InfProb = InflationProblem(dag=dag,
                           outcomes_per_party=outcomes_per_party,
                           settings_per_party=settings_per_party,
                           inflation_level_per_source=[2, 2])

InfSDP = InflationSDP(InfProb, commuting=False, verbose=1)

InfSDP.generate_relaxation(InfSDP.build_columns('local1', max_monomial_length=3))
meas = InfSDP.measurements

def prob_v(vis):
    prob = np.zeros((2,4,2,2,2))
    for a,b0,b1,c,x,z in itertools.product(range(2), range(2), range(2), range(2), range(2), range(2)):
        prob[a,2*b1+b0,c,x,z] = (1 + vis**2*(-1)**(a + c)*(((-1)**b0 + (-1)**(b1 + x + z))/2))/2**4
    return prob

# for v in np.linspace(0.85, 1., 15):
v = 1.
probability = prob_v(v)
p0 = np.sum(probability[0,:,0,0,0])
InfSDP.set_objective(meas[0][0][0][0]*meas[2][0][0][0]*meas[3][0][0][0]/p0
                     - meas[3][0][0][0])
known_values = {}
for a, b, c, x, z in itertools.product(range(1), range(3), range(1), range(2), range(2)):
    known_values[meas[0][0][x][a]*meas[1][0][0][b]*meas[2][0][z][c]] = probability[a,b,c,x,z]
for a, b, x in itertools.product(range(1), range(3), range(2)):
    known_values[meas[0][0][x][a]*meas[1][0][0][b]] = np.sum(probability[a,b,:,x,0])
for a, c, x, z in itertools.product(range(1), range(1), range(2), range(2)):
    known_values[meas[0][0][x][a]*meas[2][0][z][c]] = np.sum(probability[a,:,c,x,z])
for b, c, z in itertools.product(range(3), range(1), range(2)):
    known_values[meas[1][0][0][b]*meas[2][0][z][c]] = np.sum(probability[:,b,c,0,z])
for a, x in itertools.product(range(1), range(2)):
    known_values[meas[0][0][x][a]] = np.sum(probability[a,:,:,x,0])
for b in range(3):
    known_values[meas[1][0][0][b]] = np.sum(probability[:,b,:,0,0])
for c, z in itertools.product(range(1), range(2)):
    known_values[meas[2][0][z][c]] = np.sum(probability[:,:,c,0,z])
InfSDP.set_values(known_values)

InfSDP.solve()
print(InfSDP.primal_objective)
