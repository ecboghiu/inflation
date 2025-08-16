import einops as ep
import numpy as np

#Measurements 
e_1=1/np.sqrt(8)*np.array( [-1-1j,0,-2j,-1+1j] )
e_2=1/np.sqrt(8)*np.array( [1-1j,2j,0,1+1j] )
e_3=1/np.sqrt(8)*np.array( [-1+1j,2j,0,-1-1j] )
e_4=1/np.sqrt(8)*np.array( [1+1j,0,-2j,1-1j] )

e_1=np.transpose(e_1).conj()
e_2=np.transpose(e_2).conj()
e_3=np.transpose(e_3).conj()
e_4=np.transpose(e_4).conj()


M_1=np.outer(e_1,np.transpose(e_1.conj()))
M_2=np.outer(e_2,np.transpose(e_2.conj()))
M_3=np.outer(e_3,np.transpose(e_3.conj()))
M_4=np.outer(e_4,np.transpose(e_4.conj()))

M=[M_1,M_2,M_3,M_4]



#State
psi=1/np.sqrt(2)*np.array( [0,1,-1,0] )
rho=np.outer(psi,np.transpose(psi.conj()))
print(rho)
P0=0
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                for m in range(4):
                    for n in range(4):
                        P0=rho[i][j]*rho[k][l]*rho[m][n]*M_1[i][n]*M_1[k][j]*M_1[m][l]+P0

print(P0)
"""P=np.zeros((4,4,4))
for a in range(4):
    for b in range(4):
        for c in range(4):
            P[a,b,c]=np.einsum("ij, kl, mn, in, jk , lm -> ", rho,rho, rho, M[a], M[b], M[c])
print(P)"""
