
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

#%%
R = 10
R1 = R + 150
L = 1000
lattice = np.zeros((L, L))
lattice[L//2, L//2] = 1


N = 10**4

@jit(nopython=True)
def initial(R):
    theta = 2 * np.pi * np.random.rand()
    x = int(np.abs(L // 2 + R * np.cos(theta)))
    y = int(np.abs(L // 2 + R * np.sin(theta)))
    return np.array([x, y])


@jit(nopython=True)
def krok(x,y):
    l = [(x+1,y),(x,y+1),(x,y-1),(x-1,y)]
    l1 = []
    for l0 in l:
        if ((l0[0]>=0) and (l0[0]<L) and (l0[1]>=0) and (l0[1]<L)):
            if lattice[l0]!=1:
                l1.append(l0)
        # else:
        #     l1.append(l0)
    if len(l1)>0:
        coords2 = l1[np.random.randint(len(l1))]
    return coords2


@jit(nopython=True)
def neighbors(x,y):
    l = [(x+1,y),(x,y+1),(x,y-1),(x-1,y)]
    l1 = []
    for l0 in l:
        if ((l0[0]>=0) and (l0[0]<L) and (l0[1]>=0) and (l0[1]<L)):
            l1.append(l0)
    return l1

p = 1/2
@jit(nopython=True)
def sweep(lattice,x1,y1,R,R1,p):
    while lattice[L//2, L//2] == 1: 
        
        done=False
        l=np.sqrt((x1-L//2) ** 2 + (y1-L//2) ** 2)
        if l >= R1:
            x1, y1 = initial(R)
            print('K!')
        
        prob = np.random.random()
        for c in neighbors(x1,y1):
            if lattice[c]==1 and prob<=p:
                
                lattice[x1, y1] = 1
                R = max(R, l+10)
                R1 = R + 150

                done = True
                break
            
        if done==True:
            print('break 2')
            break
                
        else:
            x1,y1 = krok(x1, y1)
            continue
    
    return lattice,R,R1


#%%
p = 1/2
R = 10
R1 = R + 150
for i in range(N):
    print(i + 1)
    x1, y1 = initial(R)
    lattice,R,R1 = sweep(lattice,x1,y1,R,R1,p)
    if i%50==0:
        plt.title('Cząstka #'+str(i+1))
        plt.imshow(lattice)
        #plt.savefig('/home/dell/sym_komp/7-dla/ani3/'+str(i)+'.png',dpi=150) 
        plt.show()
    
#%%

plt.imshow(lattice)
#plt.savefig('/home/dell/sym_komp/7-dla/p=12.png',dpi=150) 
