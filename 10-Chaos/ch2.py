
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
#%%
f = 0.5
w = 2*np.pi*0.213

def duffing(t, xv):
    x,v = xv
    a,b,c = 1,1,0.2
    return [v, b*x - a*x**3 - c*v + f*np.cos(w*t)]
    
    
a, b = 0, 7500
t = np.arange(a,b,2*np.pi/w)


t=np.array(t)
x0 = 0
v0 = 0.4


#%%
sol1 = solve_ivp(duffing, [a, b], [x0,v0], t_eval=t)
#%%
plt.title('f='+str(f)+', x(0)='+str(x0)+', v(0)='+str(v0))
plt.xlabel('Położenie x')
plt.ylabel('Prędkość v')
#plt.plot(sol1.y[0],sol1.y[1],'o',markersize=1)

plt.scatter(sol1.y[0],sol1.y[1],s=3, c='b', lw=0,marker='o', label='punkty')

#plt.savefig('/home/dell/sym_komp/10-chaos/poincare2.png',dpi=150)


#%%

for i in range(len(sol1.y[0])):
    plt.title('f='+str(f)+', x(0)='+str(x0)+', v(0)='+str(v0))
    plt.xlabel('Położenie x')
    plt.ylabel('Prędkość v')
    plt.xlim(min(sol1.y[0])-0.1,max(sol1.y[0])+0.1)
    plt.ylim(min(sol1.y[1])-0.1,max(sol1.y[1])+0.1)
    plt.scatter(sol1.y[0][:i],sol1.y[1][:i],s=3, c='b', lw=0,marker='o', label='punkty')
    #plt.savefig('/home/dell/sym_komp/10-chaos/poincare/'+str(i)+'.png',dpi=150)
    plt.show()
    plt.clf()


