import numpy as np
import random
import matplotlib.pyplot as plt
#%%

L = 31
hc = 3
tmax = 50000
s = 400

grid = np.zeros( (L, L), int)

rozmiary = [] #= np.empty(tmax-s,int)



for i in range(tmax):
    
    if i%1000:
        print(i)
    
    lawiny = np.zeros( (L, L), int)
    
    while np.max(grid) <= hc:
        
        (x,y) = (random.randint(1,L-2),random.randint(1,L-2))
        grid[(x,y)] += 1
    
    while np.max(grid) > hc:
        ix,iy = np.where(grid > hc)
        grid[(ix,iy)] -= 4
        grid[(ix+1,iy)] += 1
        grid[(ix-1,iy)] += 1
        grid[(ix,iy+1)] += 1
        grid[(ix,iy-1)] += 1
        
        # empty border
        grid[0,:] = 0
        grid[:,0] = 0
        grid[L-1,:] = 0
        grid[:,L-1] = 0
        
        
        for j in range(len(ix)):
            
            if not lawiny[(ix[j],iy[j])] > 0:
                lawiny[(ix[j],iy[j])] += 1
    
    if i>s:
        rozmiary.append(np.sum(lawiny))
    
    
    
#%%


hist, bins = np.histogram( rozmiary, bins = np.max(rozmiary), density=False)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Rozmiar lawiny')
plt.ylabel('Częstość')

X = bins[0:-1]
Y = hist
plt.plot(bins[0:-1],hist, label='Symulacja')

from scipy.optimize import curve_fit

def f(x,a,b):
    return a/x + b

guess = [1000,1]

par, cov = curve_fit(f, X,Y, guess)
xfit=np.arange(min(X),max(X),1)

plt.plot(xfit, f(xfit, *par), color='r', linewidth=0.75, label='Dopasowanie')

plt.legend()

#plt.savefig('/home/dell/sym_komp/11-bak/zad2_2.png',dpi=150) 
plt.show()



