import numpy as np
import matplotlib.pyplot as plt
from random import choice

#%% Fractal 1
n=3

M1 = np.array([0.5, 0, 0, 0.5, 0.25, np.sqrt(3.)/4])
M2 = np.array([0.5, 0, 0, 0.5, 0.0, 0])
M3 = np.array([0.5, 0, 0, 0.5, 0.5, 0])

P = [1/3,1/3,1/3]

#%% Fractal 2
n = 4

P = [0.02,0.09,0.10,0.79]
M1 = np.array([0.001, 0.0, 0.0, 0.16, 0.0, 0.0])
M2 = np.array([-0.15, 0.28, 0.26, 0.24, 0.0, 0.44])
M3 = np.array([ 0.2,-0.26, 0.23, 0.22, 0.0, 1.6])
M4 = np.array([ 0.85, 0.04,-0.04, 0.85, 0.0, 1.6])


#%%

N = 20000
x,y = 0,0
xs,ys = np.empty(N),np.empty(N)

for i in range(N):
    a = np.random.choice(n,p=P)

    if a==0:
        M = M1
    if a==1:
        M = M2
    if a==2:
        M = M3
    if a==3:
        M = M4
        
    m0,m1,m2,m3,m4,m5 = M

    x1 = m0*x + m1*y + m4
    y1 = m2*x + m3*y + m5
    
    xs[i] = x1
    ys[i] = y1
    
    x = x1
    y = y1
    
#%%

#plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['axes.facecolor'] = 'black'
plt.scatter(xs,ys, s=1, marker="o", lw=0,c='green')
plt.savefig('Fractal2.png', dpi=150)
