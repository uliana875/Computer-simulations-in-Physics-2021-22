import numpy as np
import matplotlib.pyplot as plt

G = 0.01
M = 500
m = 0.1
dt = 0.001


r0 = np.array([2,0])
p0 = np.array([0,0.1])
F0 = -(G*M*m*r0)/(r0[0]**2 + r0[1]**2)**(3/2)


N = 10000

rs = np.empty((N+2,2))
rs[0][0] = r0[0]
rs[0][1] = r0[1]

ps = np.empty((N+2,2))
ps[0][0] = p0[0]
ps[0][1] = p0[1]


r1 = r0 + (p0*dt)/m + (F0*dt**2)/(2*m)
p1 = p0 + F0*dt
F1 = -(G*M*m*r1)/(r1[0]**2 + r1[1]**2)**(3/2)

rs[1][0] = r1[0]
rs[1][1] = r1[1]

pot = [-(G*m*M)/np.linalg.norm(r0), -(G*m*M)/np.linalg.norm(r1)]

#%% Verlet
for i in range(N):
    r2 = 2*r1 - r0 + (F1*dt**2)/m
    F2 = -(G*M*m*r2)/(r2[0]**2 + r2[1]**2)**(3/2)
    
    rs[i+2][0] = r2[0]
    rs[i+2][1] = r2[1]
    
    r0 = r1
    r1 = r2
    F1 = F2
    
    pot.append(-(G*m*M)/np.linalg.norm(r2))
    
xs = rs[:,0]
ys = rs[:,1]


vxs = np.gradient(xs)/dt
vys = np.gradient(ys)/dt

kin = (m*(vxs**2 + vys**2))/2

E = kin + pot

plt.plot(E,label='E')
plt.plot(pot,label='U')
plt.plot(kin,label='K')

plt.xlabel('Step')
plt.ylabel('Energy')
plt.legend()
plt.savefig('Verlet_energy.png', dpi=150)
plt.show()

plt.plot(xs,ys)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Verlet_orbit.png', dpi=150)
plt.show()



