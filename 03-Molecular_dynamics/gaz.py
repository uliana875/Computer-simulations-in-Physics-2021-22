import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#%%

sig = 1
eps = 1 
rc = 2.5*sig

dx = 2
dy = 2

nx = 4
ny = 4

N = nx*ny

radius = 0.5

time = np.arange(5000)

Temp = 2.5
k = 1 


m = 1

dt = 0.0001

boxsize = 8
promien = 0.5

class czastka:
    def __init__(self,promien,pos,vel,his,his_vel):
        self.promien = promien
        self.r = pos
        self.v = vel
        self.his = his
        self.hisv = his_vel
        
particles = []


# tworzymy cząstki
for i in range(nx):
    for j in range(ny):
        pol = np.array([i*dx+1,j*dy+1])
        vel = np.array([random.uniform(0,1), random.uniform(0,1)])
        particles.append(czastka(radius, pol, vel,np.empty((len(time),2)),np.empty((len(time),2))))


# usuwamy prędkość śm
sumv = 0.0
for p in particles:
    sumv += p.v

av_v = sumv/N

for p in particles:
    p.v = p.v - av_v
    

# skalujemy vs do T
sumv2 = 0.0
for p in particles:
    sumv2 +=np.dot(p.v,p.v)/2.0
    
av_v2 = sumv2/N
fs = np.sqrt(Temp/av_v2)

for p in particles:
    p.v = p.v*fs



def closest_image(x1,x2):
    x12 = x2 - x1
    if x12[0] > boxsize / 2:
        x12[0] = x12[0] - boxsize
    elif x12[0] < -boxsize / 2:
        x12[0] = x12[0] + boxsize
    if x12[1] > boxsize / 2:
        x12[1] = x12[1] - boxsize
    elif x12[1] < -boxsize / 2:
        x12[1] = x12[1] + boxsize
    return x12


def bound(new_r):
    if new_r[0] > boxsize:
        new_r[0] = new_r[0] - boxsize
    if new_r[0] < 0:
        new_r[0] = new_r[0] + boxsize
        
    if new_r[1] > boxsize:
        new_r[1] = new_r[1] - boxsize
    if new_r[1] < 0:
        new_r[1] = new_r[1] + boxsize
        
    return new_r



def F(r1,r2):
    new_r = closest_image(r1,r2)
    r12 = np.sqrt(new_r[0]**2 + new_r[1]**2)
    if r12<=rc:
        f = (48*eps/sig)*((sig/r12)**13 - (1/2)*(sig/r12)**7)  #
    else: 
        f = 0
    return -(f*new_r)/r12

def potencjal(r1,r2):
    x12 = closest_image(r1,r2)
    r12 = np.sqrt(x12[0]**2 + x12[1]**2)
    if r12<=rc:
        u = 4*eps*( (sig/r12)**12 - (sig/r12)**6) - 4*eps*( (sig/rc)**12 - (sig/rc)**6) #last -
    else: 
        u = 0
    return u
    

F0 = []
for pi in particles:
    f0 = 0
    for pj in particles:
        if not pi==pj:
            f0 += F(pi.r,pj.r)
    F0.append(f0)
    
    
v1 = [pi.v + (F0[particles.index(pi)]*dt)/m for pi in particles]       

    

for i in range(len(particles)):
    particles[i].v = (particles[i].v + v1[i])/2

#%%

T = np.empty(len(time))
P = np.empty(len(time))
V = boxsize**2
E_kin,E_pot,E = np.empty(len(time)),np.empty(len(time)),np.empty(len(time))

for i in range(len(time)):
    K = 0
    fr = 0
    U = 0
    for pi in particles: 
        
        F1 = 0
        for pj in particles:
            if not pi==pj:
               f1 = F(pi.r,pj.r)
               F1 += f1
               
               r12 = closest_image(pi.r,pj.r)
               fr += 0.5*np.dot(r12,f1)
               U += 0.5*potencjal(pi.r,pj.r)
               
           

        v12 = pi.v + (F1*dt)/m  
        
        v1 = (pi.v + v12)/2
        K += (m/2)*(v1[0]**2 + v1[1]**2)
        new = pi.r + v12*dt

        new_r = bound(new)

        pi.r = new_r
        pi.v = v12
       
        pi.his[i,:] = pi.r
        pi.hisv[i,:] = pi.v
    
    T[i] = K/N
    p = K/V - fr/(2*V)
    P[i] = p
    
    E_kin[i] = K
    E_pot[i] = U
    E[i] = K + U
    
    #animacja
    if (i%100==0):
        plt.clf()
        fig= plt.gcf()
        for p in particles:
            a = plt.gca()
            cir= Circle((p.r[0],p.r[1]), radius=p.promien) 
            a.add_patch(cir)
        plt.plot()
        plt.xlim((0,boxsize))
        plt.ylim((0,boxsize))    
        fig.set_size_inches((6,6))
        #plt.savefig('/home/dell/sym_komp/3-dynamika_molekularna/ani/img'+str(i)+'.png') 
        plt.show()

       

#%%

plt.plot(P,label='P')
plt.plot(T,label='T')
plt.plot(E,label='E')
plt.xlabel('Krok')
plt.legend()
#plt.savefig('P,T.png',dpi=150)
plt.show()
#%%

plt.plot(E,label='E')
plt.xlabel('Krok')
plt.legend()
plt.savefig('E.png',dpi=150)
plt.show()
#%%

T = np.empty(len(time))

for i in range(len(time)):
    K = 0
    for pi in particles:
        K += m*(pi.hisv[i][0]**2 + pi.hisv[i][1]**2)/2
    temp = K/16 ###
    T[i] = temp
    
plt.title('Temperatura')    
plt.plot(T)
plt.savefig('T.png',dpi=150)
plt.show()
    
#%%

# close field

P = np.empty(len(time))
V = 8*8

for i in range(len(time)):
    print(i)
    fr = 0
    for pi in particles:
        for pj in particles:
            if not pi==pj:
               f = F(pi.his[i],pj.his[i])
               r12 = closest_image(pi.his[i],pj.his[i])
               fr += 0.5*np.dot(r12,f) 
    p = (16*T[i])/V - fr/(2*V)
    P[i] = p
#%%

plt.plot(E_pot, label='U')
plt.plot(E_kin, label='K')
plt.plot(E,label='E')
plt.legend()
plt.savefig('Energy.png',dpi=150)