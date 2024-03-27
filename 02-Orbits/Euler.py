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

rs = np.empty((N+1,2))
rs[0][0] = r0[0]
rs[0][1] = r0[1]

ps = np.empty((N+1,2))
ps[0][0] = p0[0]
ps[0][1] = p0[1]

pot = [ -(G*m*M)/np.linalg.norm(r0)]

#%% Euler


for i in range(N):
    r1 = r0 + (p0*dt)/m + (F0*dt**2)/(2*m)
    p1 = p0 + F0*dt
    F1 = -(G*M*m*r1)/(r1[0]**2 + r1[1]**2)**(3/2)
    pot.append( -(G*m*M)/np.linalg.norm(r1))
    rs[i+1][0] = r1[0]
    rs[i+1][1] = r1[1]
    ps[i+1][0] = p1[0]
    ps[i+1][1] = p1[1]
    
    r0 = r1
    p0 = p1
    F0 = F1
    

kin = (ps[:,0]**2 + ps[:,1]**2)/(2*m)

plt.plot(kin)

E = kin + pot

plt.plot(E)
print(E)


plt.plot(pot)
plt.show()

xs = rs[:,0]
ys = rs[:,1]

plt.plot(xs,ys)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Euler_orbit.png', dpi=150)
plt.show()