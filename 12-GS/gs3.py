import numpy as np
import matplotlib.pyplot as plt

N=100
Du = 2e-5
Dv = 1e-5
F = np.arange(0,0.1,0.005).tolist()
k = 0.062
dx,dy = 0.02,0.02
dt=1

u = np.ones((N,N))
v = np.zeros((N,N))
N1=N//4
N2=3*N//4
u[N1:N2+1, N1:N2+1] = 0.4 + np.random.rand(N2-N1+1, N2-N1+1) * 0.2
v[N1:N2+1, N1:N2+1] = 0.2 + np.random.rand(N2-N1+1, N2-N1+1) * 0.2

T = 10000

def lap_2D(f):
    f_rx = np.roll(f,-1,axis=1)
    f_lx = np.roll(f,1,axis=1)    
    f_ry = np.roll(f,-1,axis=0)    
    f_ly = np.roll(f,1,axis=0)  
    return (f_rx + f_lx + f_ry + f_ly - 4*f) / dx**2

def du_dt(u,v,F):
    return Du*lap_2D(u) - u*v**2 + F*(1-u)

def dv_dt(u,v,F):
    return Dv*lap_2D(v) + u*v**2 - (F+k)*v



U, V = np.empty((T,N)),np.empty((T,N))

for f in F:
    
    u = np.ones((N,N))
    v = np.zeros((N,N))
    N1=N//4
    N2=3*N//4
    u[N1:N2+1, N1:N2+1] = 0.4 + np.random.rand(N2-N1+1, N2-N1+1) * 0.2
    v[N1:N2+1, N1:N2+1] = 0.2 + np.random.rand(N2-N1+1, N2-N1+1) * 0.2
    
    for i in range(T):
        u1 = u + du_dt(u,v,f)*dt
        v1 = v + dv_dt(u,v,f)*dt
        
        u,v = u1,v1
        
        
    plt.title('F='+str(f)+', k='+str(k))
    plt.imshow(v)
    #plt.savefig('/home/dell/sym_komp/12-gs/F='+str(f)+'.png',dpi=150)
    plt.show()