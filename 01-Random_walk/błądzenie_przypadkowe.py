import numpy as np
import random
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20,10)
#%% 1 cząstka

T = 1000
dx = np.random.randn(T)

x = np.cumsum(dx) 

t = np.arange(0,T)

plt.title('Położenie cząstki w czasie')
plt.xlabel('Czas t')
plt.ylabel('Położenie x')
plt.plot(t,x)
plt.savefig('Particle_position_over_time.png', dpi=150)

#%% N cząstek

# dx - to macierz
T = 100
N = 10

dx = np.random.randn(N,T)

x = np.cumsum(dx,axis=1)

t = np.arange(0,T)

#%% pętla do rysowania
plt.rcParams['figure.figsize'] = (15,10)
cmap=plt.get_cmap('Set1')
opis = [] # <- tablica na opisy

for i in range(N):
    plt.plot(t, x[i,:], color=cmap(i), alpha=0.7)
    opis.append('Cząstka: {}'.format(i+1))


plt.title('Położenie cząstek w czasie')
plt.xlabel('Czas t')
plt.ylabel('Położenie x')
plt.legend(opis,prop={'size': 12})
plt.savefig('N_particles_positions_over_time.png', dpi=150)

#%% 2D

dy = np.random.randn(N,T)

y = np.cumsum(dy,axis=1)

#%%

for i in range(N):
    plt.plot(y[i,:], x[i,:], color=cmap(i), alpha=0.7)
    opis.append('Cząstka: {}'.format(i+1))


plt.title('Trajektorie cząstek')
plt.xlabel('y')
plt.ylabel('x')
plt.legend(opis,prop={'size': 12})

plt.savefig('particles_trajectories.png', dpi=150)
#%% równanie dyfuzji

T = 10
N = 100

dt = 1
D = 1

dx = np.sqrt(2*D*dt)*np.random.randn(N,T)
x_end = np.sort( np.sum(dx,axis=1) )

t = np.arange(0,T)

def rho(x,t):
    return (1/np.sqrt(4*np.pi*D*t))*np.exp(-x**2/(4*D*t)) 
    
plt.plot(x_end,rho(x_end,T),linewidth=5)


hist = plt.hist(x_end, bins=50, edgecolor = "black", linewidth=0.5,density=True)

plt.xlabel('Końcowe położenie')
plt.ylabel('Gęstość prawdopodobieństwa')
plt.savefig('diffusion_equation.png', dpi=150)

#%%

T = 10
N = 8000

dt = 1
D = 1

dx = np.sqrt(2*D*dt)*np.random.randn(N,T)

x_end = np.cumsum(dx,axis=1)

t = np.arange(1,T+1)
#%%

for i in range(T):
    xx = np.sort(x_end[:,i])
    plt.xlim(-16,16)
    plt.ylim(0,0.3)
    plt.xlabel('Końcowe położenie')
    plt.ylabel('Gęstość prawdopodobieństwa')
    plt.hist(xx, bins=50, edgecolor = "black", linewidth=0.5,density=True)
    plt.plot(xx,rho(xx,t[i]),'o')
    plt.show()
    
#%% dodatkowe

T = 50
N = 100

dt = 1
D = 10

dx = np.sqrt(2*D*dt)*np.random.randn(N,T)

x_end = np.cumsum(dx,axis=1)

t = np.arange(1,T+1)
#%%

msds = []
for i in range(T):
    msd = np.sum ( (x_end[:,i])**2/N )
    msds.append(msd)
#%%
plt.plot(t,msds,label='Symulacja')
plt.plot(t,2*D*t,label='Analityczny')
plt.legend(prop={'size': 20})
plt.xlabel('Czas')
plt.ylabel('Średnia')
plt.title('Średnie odchylenie kwadratowe położeń cząstek (N=10000)')
plt.savefig('simulation_vs_formula.png', dpi=150)