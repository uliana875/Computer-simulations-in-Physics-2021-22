import numpy as np
import matplotlib.pyplot as plt
#%%

L = 100
hc = 3
tmax = 3000

grid = np.ones( (L, L), int)*7
grid[0,:] = 0
grid[:,0] = 0
grid[L-1,:] = 0
grid[:,L-1] = 0


i=0
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
    

    if (i%30==0):
        plt.title('Krok #'+str(i))
        plt.imshow(grid, cmap='rainbow', vmin=0, vmax=8)
        plt.savefig('/home/dell/coding_subjects/sym_komp/11-bak/3/img'+str(i)+'.png') 
        plt.show()
        
    i+=1
    
    
plt.title('50000')
plt.imshow(grid, cmap='rainbow', vmin=0, vmax=8)