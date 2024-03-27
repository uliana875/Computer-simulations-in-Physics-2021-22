import numpy as np
import random
from numba import jit
import matplotlib.pyplot as plt

L = 500
N = L**2
J = 1




@jit(nopython=True)
def neighbors(x,y):
    a = [x+1,y]
    b = [x,y+1]
    c = [x-1,y]
    d = [x,y-1]
    l = [a,b,c,d]
    l1 = []
    for l0 in l:
        if l0[0]>=L:
            l0[0]=0
            
        if l0[0]<0:
            l0[0]=L-1
            
        if l0[1]>=L:
            l0[1]=0
            
        if l0[1]<0:
            l0[1]=L-1

        l1.append((l0[0],l0[1]))
    
    return l1


#%%




@jit(nopython=True)
def sweep(lattice):
    for i in range(L**2):
        x = np.random.randint(L)
        y = np.random.randint(L)
        
        sn = 0
        for sj in neighbors(x,y):
            sn += lattice[sj]
            
        delta = 2*J*sn*lattice[x][y]
        
        if delta<=0:
            lattice[x][y] = -lattice[x][y] 
        else:       
            r = random.uniform(0,1)
            if r < np.exp(-b*delta):
                lattice[x][y] = -lattice[x][y]
                
        #mag = np.mean(lattice)
    return lattice


    
#%%#
lattice = np.ones((L,L), dtype=int)

for i in range(L):
    for j in range(L):
        lattice[i][j] = random.choice([1,-1])*lattice[i][j]
plt.imshow(lattice)
#plt.savefig('/home/dell/sym_komp/9-metropolis/domeny-0.png',dpi=150)
#%%


T = 2
b = 1/T

czasy = [9,19,49,99,199,499,999,1999,4999]
Rs = []

for i in range(5000):
    print(i)
    lattice = sweep(lattice)
    if i in czasy:
        chi = np.empty(L//2)
        
        for r in range(L//2):
            suma = 0
            for r0 in range(L):
                if r0+r>=L:
                    k = (r0+r)%L
                else:
                    k = r0+r
                suma += lattice[:,r0]*lattice[:,k]/L    
            chi[r] = np.mean(suma)
        X = np.arange(0,L//2,1)    
        chi2 = chi[np.argwhere(chi>0.3)]
        C = -np.sum( np.log(chi2) )
        rmax = X[len(chi2)-1]
        R = (rmax*(rmax+1))/(2*C)
        Rs.append(R)
        
        
    if i==9 or i==49 or i==99 or i==999 or i==4999:
        plt.title(str(i+1)+' kroków')
        plt.imshow(lattice)
        #plt.savefig('/home/dell/sym_komp/9-metropolis/domeny-'+str(i+1)+'.png',dpi=150)
        plt.show()
        
        plt.title(str(i+1)+' kroków')
        plt.xlabel('r')
        plt.ylabel('Chi (r)')
        plt.plot(X,chi)
        #plt.savefig('/home/dell/sym_komp/9-metropolis/chi-'+str(i+1)+'.png',dpi=150)
        plt.show()
        
        

#%%

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Czas t')
plt.ylabel('Rozmiar domen R')
plt.plot(czasy, Rs,'o')
plt.plot([czasy[0],czasy[-1]],[Rs[0],Rs[-1]])
#plt.savefig('/home/dell/sym_komp/9-metropolis/rozmiar-domen.png',dpi=150)


