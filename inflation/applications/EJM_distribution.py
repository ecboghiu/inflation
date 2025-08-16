import numpy as np
P_NS=np.zeros((4,4,4))

for a in range(4):
    P_NS[a, a, a] = 1 / 8
    for b in range(4):
        for c in range(4):
            if a!=b and b!=c and c!=a:
                P_NS[a,b,c]=1/48


print("1/P_NS(A=i,B=i)=",1/P_NS[0, 0, :].sum())
print("1/P_NS(A=i,B=j)=",1/P_NS[0, 1, :].sum())
print("1/P_NS(A=i,B=j)=", 1/((1-4*P_NS[0, 0, :].sum())/12))
print("1/P_NS(000): ",1/P_NS[0,0,0])
print("P_NS(001): ", P_NS[0,0,1])
print("1/P_NS(012): ",1/P_NS[0,1,2])


print("Normalization test: ",P_NS.sum())

EJM = 3/4 * P_NS + (1/4)*(1/64)
print("EJM recovery (000): ",256*EJM[0,0,0])
print("EJM recovery (001): ",256*EJM[0,0,1])
print("EJM recovery (012): ",256*EJM[0,1,2])