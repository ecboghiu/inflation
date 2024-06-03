from inflation import InflationProblem, InflationSDP


scenario = InflationProblem(dag={'s': ['A', 'B']},
                            outcomes_per_party=[4, 4],
                            settings_per_party=[2, 2],
                            inflation_level_per_source=(1,),
                            classical_sources=None,
                            order=['A','B'],
                            verbose=2)
sdp = InflationSDP(scenario, verbose=2)
sdp.generate_relaxation("npa3")
print(sdp.knowable_atoms)

objective_dict = {
    "P[A_0=0]": -1,
    "P[A_0=1]": -1,

    "P[B_0=0]": -1,
    "P[B_0=1]": -1,

    "P[A_0=0 B_0=1]": 1,
    "P[A_0=1 B_0=0]": 1,

    "P[A_1=0 B_0=1]": 1,
    "P[A_1=1 B_0=0]": 1,
    "P[A_1=2 B_0=0]": 1,
    "P[A_1=2 B_0=1]": 1,

    "P[A_0=1 B_1=0]": 1,
    "P[A_0=0 B_1=1]": 1,
    "P[A_0=0 B_1=2]": 1,
    "P[A_0=1 B_1=2]": 1,

    "P[A_1=1 B_1=0]": -1,
    "P[A_1=0 B_1=1]": -1,
    "P[A_1=0 B_1=2]": -1,
    "P[A_1=2 B_1=0]": -1,
    "P[A_1=1 B_1=2]": -1,
    "P[A_1=2 B_1=1]": -1,
    "P[A_1=2 B_1=2]": -1
}

sdp.set_objective(objective_dict, direction='max')
sdp.solve()
print(sdp.primal_objective)