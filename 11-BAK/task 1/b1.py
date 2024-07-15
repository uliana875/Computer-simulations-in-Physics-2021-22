import numpy as np
import random
import matplotlib.pyplot as plt
#%%

L = 31
hc = 3
tmax = 1000

grid = np.zeros( (L, L), int)

num = np.empty(tmax)

for i in range(tmax):
    
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
        
    num[i] = np.sum(grid)
    
    if (i%50==0):
        plt.title('Krok #'+str(i))
        plt.imshow(grid, cmap='rainbow', vmin=0, vmax=8)
        #plt.savefig('/home/dell/sym_komp/11-bak/img'+str(i)+'.png') 
        plt.show()
    
t = np.arange(tmax)

plt.xlabel('Czas')
plt.ylabel('Liczba ziaren')
plt.plot(t,num)
#plt.savefig('/home/dell/sym_komp/11-bak/wykres1.png') 

#%%

# dodawanie ziarna na środek

L = 31
hc = 3
tmax = 1500

grid = np.zeros( (L, L), int)

num = np.empty(tmax)

for i in range(tmax):
    
    while np.max(grid) <= hc:

        grid[(L//2,L//2)] += 1
    
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
        
    num[i] = np.sum(grid)
    
    if (i%50==0):
        plt.title('Krok #'+str(i))
        plt.imshow(grid, cmap='rainbow', vmin=0, vmax=8)
        #plt.savefig('/home/dell/sym_komp/11-bak/img'+str(i)+'.png') 
        plt.show()
    
t = np.arange(tmax)

plt.xlabel('Czas')
plt.ylabel('Liczba ziaren')
plt.plot(t,num)
#plt.savefig('/home/dell/sym_komp/11-bak/wykres2.png') 