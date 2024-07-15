from scipy import linspace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
#%%

def lorenz(t, xyz):
    x, y, z = xyz
    s, r, b = 10, 28, 8/3. # parameters Lorentz used
    return [s*(y-x), x*(r-z) - y, x*y - b*z]

a, b = 0, 40
t = linspace(a, b, 4000)

sol1 = solve_ivp(lorenz, [a, b], [1,1,1], t_eval=t)
sol2 = solve_ivp(lorenz, [a, b], [1,1,1.00001], t_eval=t)

plt.plot(sol1.y[0], sol1.y[2])
plt.xlabel("$x$")
plt.ylabel("$z$")
#plt.savefig("lorenz_xz.png")
plt.close()

plt.subplot(211)
plt.plot(sol1.t, sol1.y[0])
plt.xlabel("$t$")
plt.ylabel("$x_1(t)$")
plt.subplot(212)
plt.plot(sol1.t, sol1.y[0] - sol2.y[0])
plt.ylabel("$x_1(t) - x_2(t)$")
plt.xlabel("$t$")
#plt.savefig("lorenz_x.png")

#%%
f = 0.18


def duffing(t, xv):
    x,v = xv
    a,b,c,w = 1,1,0.2,2*np.pi*0.213
    return [v, b*x - a*x**3 - c*v + f*np.cos(w*t)]
    
    
a, b = 0, 500
t = np.linspace(a, b, 5000)

x0 = 1.75
v0 = 0.78

sol1 = solve_ivp(duffing, [a, b], [x0,v0], t_eval=t)

plt.title('f='+str(f)+', x(0)='+str(x0)+', v(0)='+str(v0))
plt.xlabel('Położenie x')
plt.ylabel('Prędkość v')
plt.plot(sol1.y[0][250:],sol1.y[1][250:])
#plt.savefig('/home/dell/sym_komp/10-chaos/3_okresy.png',dpi=150)
#%%
plt.xlabel('Czas t')
plt.ylabel('Położenie x')
plt.plot(t,sol1.y[0])
plt.show()

plt.xlabel('Czas t')
plt.ylabel('Prędkość v')
plt.plot(t,sol1.y[1])
plt.show()


#%%