import numpy as np
from collections import deque
import matplotlib.pyplot as plt

#%%

AAA,PPP = [],[]

#%%


def neighbors(i):
    (x,y) = i
    a = (x+1,y)
    b = (x,y+1)
    c = (x-1,y)
    d = (x,y-1)
    l = [a,b,c,d]
    l1 = []
    for l0 in l:
        if ((l0[0]>=0) and (l0[0]<L) and (l0[1]>=0) and (l0[1]<L)):
            l1.append(l0)
    return l1




probabilities = [0.15,0.25,0.35,0.45,0.48,0.51,0.55,0.58,0.59,0.61,0.65,0.75,0.85]

Prob,Area = [],[]

for pi in probabilities:
    
    N = 200
      
    n_edge = 0
    s_num = 0
    S = 0
    
    for k in range(N):
        L = 200
        lattice = - np.ones((L,L))
    
        p = pi
        tossing = np.random.random((L,L)) < p
    
    
        i = (L//2,L//2)
        lattice[i] = 1
        cluster = deque()
    
        cluster.append(i)
    
        
        while not (len(cluster)==0):
            for j in neighbors(i):
                if (lattice[j]==-1):
                    if (tossing[j]):
                        cluster.append(j)
                        lattice[j]=1
                    else:
                        lattice[j]=0
            cluster.popleft()
            if not (len(cluster)==0):
                i = cluster[0]

        
        if ((np.all(lattice[0,:]==-1)==False) or (np.all(lattice[L-1,:]==-1)==False) or
            (np.all(lattice[:,0]==-1)==False) or (np.all(lattice[:,L-1]==-1)==False) ):
            n_edge += 1
            
        if ((np.all(lattice[0,:]==-1)==True) and (np.all(lattice[L-1,:]==-1)==True) and
            (np.all(lattice[:,0]==-1)==True) and (np.all(lattice[:,L-1]==-1)==True) ):
            S += len(np.argwhere(lattice>=0))
            s_num += 1
            print(s_num)
            
    Prob.append(n_edge/N)
    if not s_num==0:
        Area.append(S/s_num)
    else:
        Area.append(0)
        
AAA.append(Area)
PPP.append(Prob)        
            
#%%

fig, axs = plt.subplots(2)

fig.text(0.5, 0.04, 'p', ha='center')
fig.suptitle('Rozmiar sieci: '+str(L)+'x'+str(L))
axs[0].set_ylabel('P(p)')
axs[1].set_ylabel('S(p)')

axs[0].plot(probabilities,Prob,'o',color='darkorange')
axs[1].plot(probabilities,Area,'o',color='forestgreen')
#plt.savefig('L='+str(L)+'2.png', dpi = 150)

