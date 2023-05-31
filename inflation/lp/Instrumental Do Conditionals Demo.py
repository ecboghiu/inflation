from inflation import InflationProblem, InflationLP, InflationSDP
import numpy as np

p = np.zeros((2, 2, 2, 1))
p[0, 0, 0, 0] = 0.3
p[1, 0, 0, 0] = 0.7
p[0, 0, 1, 0] = 0.7
p[1, 1, 1, 0] = 0.3
p = 0.9 * p + 0.1 * (1 / 4)




instrumental = InflationProblem({"U_AB": ["A", "B"],
                                 "A": ["B"]},
                                outcomes_per_party=(2, 2),
                                settings_per_party=(2, 1),
                                inflation_level_per_source=(1,),
                                order=("A", "B"),
                                classical_sources=["U_AB"])
print("Now considering FANOUT inflation (a.k.a. `Unpacking`)")
instrumental_infLP = InflationLP(instrumental,
                                 nonfanout=False,
                                 verbose=False)
instrumental_infLP.generate_lp()
instrumental_infLP.set_distribution(p)
bounds = dict()

instrumental_infLP.set_objective(objective={'<B_1_0_1>': 1},
                                 direction='min')
instrumental_infLP.solve()
bounds["P(Y=1 do X#=0) >= "] = instrumental_infLP.objective_value

instrumental_infLP.set_objective(objective={'<B_1_0_1>': 1},
                                 direction='max')
instrumental_infLP.solve()
bounds["P(Y=1 do X#=0) <= "] = instrumental_infLP.objective_value

instrumental_infLP.set_objective(objective={'<B_1_1_1>': 1},
                                 direction='min')

instrumental_infLP.solve()
bounds["P(Y=1 do X#=1) >= "] = instrumental_infLP.objective_value

instrumental_infLP.set_objective(objective={'<B_1_1_1>': 1},
                                 direction='max')
instrumental_infLP.solve()
bounds["P(Y=1 do X#=1) <= "] = instrumental_infLP.objective_value
for k, v in bounds.items():
    print(f"{k}{v:.3f}")


print("Now considering NONFANOUT inflation (a.k.a. `SWIGs\INTERRUPTION`)")
instrumental_infLP = InflationLP(instrumental,
                                 nonfanout=True,
                                 verbose=False)
instrumental_infLP.generate_lp()
instrumental_infLP.set_distribution(p)

bounds = dict()

instrumental_infLP.set_objective(objective={'<B_1_0_1>': 1},
                                 direction='min')
instrumental_infLP.solve()
bounds["P(Y=1 do X#=0) >= "] = instrumental_infLP.objective_value

instrumental_infLP.set_objective(objective={'<B_1_0_1>': 1},
                                 direction='max')
instrumental_infLP.solve()
bounds["P(Y=1 do X#=0) <= "] = instrumental_infLP.objective_value

instrumental_infLP.set_objective(objective={'<B_1_1_1>': 1},
                                 direction='min')

instrumental_infLP.solve()
bounds["P(Y=1 do X#=1) >= "] = instrumental_infLP.objective_value

instrumental_infLP.set_objective(objective={'<B_1_1_1>': 1},
                                 direction='max')
instrumental_infLP.solve()
bounds["P(Y=1 do X#=1) <= "] = instrumental_infLP.objective_value
for k, v in bounds.items():
    print(f"{k}{v:.3f}")