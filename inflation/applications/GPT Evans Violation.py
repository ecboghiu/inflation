from inflation import InflationProblem, InflationLP
import numpy as np

p_Q1=np.zeros((2,2,3,1,1,1),dtype=float)
q=[8/17,8/17,1/17]
p_Q1_do=np.zeros((2,2,3),dtype=float)
#v=1/np.sqrt(2)
v=1
def d(x,y):
    if x==y :
        return 1
    elif x !=y :
        return 0
for a in range(2):
    for b in range(2):
        for y in range(3):
            if y==2:
                p_Q1_do[a,b,2]=q[2]*(1/2)
                p_Q1[a,b,2]=(q[2])*1/2*d(b,0)
            else:
                p_Q1_do[a,b,y]=q[y]*(1/2)
                p_Q1[a,b,y]=(q[y])*(1/4)*((1+(v)*((-1)**(a+b+b*y))))

Evans = InflationProblem({"U_AB": ["A", "B"],
                          "U_BC": ["B", "C"],
                          "B": ["A", "C"]},
                         outcomes_per_party=(2, 2, 3),
                         settings_per_party=(1, 1, 1),
                         inflation_level_per_source=(2, 2),  # TO BE MODIFIED
                         order=("A", "B", "C"),
                         verbose=2)

Evans_Unpacked = InflationLP(Evans,
                             nonfanout=False)
Evans_Unpacked.set_distribution(p_Q1)
Evans_Unpacked.solve(dualise=True, verbose=2)
print(Evans_Unpacked.status)
print(Evans_Unpacked.solution_object['x'])
print(Evans_Unpacked.certificate_as_probs())


