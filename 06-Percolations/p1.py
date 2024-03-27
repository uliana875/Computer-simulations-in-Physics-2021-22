import numpy as np
from collections import deque
import matplotlib.pyplot as plt

#%%

L = 20
lattice = - np.ones((L,L))

p = 0.65
tossing = np.random.random((L,L)) < p


i = (L//2,L//2)
lattice[i] = 1
cluster = deque()

cluster.append(i)



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

  

while not (len(cluster)==0):
    for j in neighbors(i):
        if (lattice[j]==-1):
            if (tossing[j]):
                cluster.append(j)
                lattice[j]=1
            else:
                lattice[j]=0
    print(cluster)
    cluster.popleft()
    if not (len(cluster)==0):
        i = cluster[0]
                

                
plt.imshow(lattice,interpolation='nearest',cmap='magma')
plt.grid()

#%%

L = 100
lattice = - np.ones((L,L))

p = 0.59
tossing = np.random.random((L,L)) < p


i = (L//2,L//2)
lattice[i] = 1
cluster = deque()

cluster.append(i)

  
k = 0
while not (len(cluster)==0):
    for j in neighbors(i):
        if (lattice[j]==-1):
            if (tossing[j]):
                cluster.append(j)
                lattice[j]=1
            else:
                lattice[j]=0
    #print(cluster)
    cluster.popleft()
    if not (len(cluster)==0):
        i = cluster[0]
    if k%100==0:
        plt.imshow(lattice,interpolation='nearest',cmap='magma')
        plt.grid()
        plt.savefig(str(k)+'.png', dpi=150)
        plt.show()
    k+=1
    #print(k)

                