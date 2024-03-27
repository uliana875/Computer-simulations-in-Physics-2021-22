import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import moviepy.video.io.ImageSequenceClip
#%% consts

sig = 1
eps = 1 
rc = 2.5*sig

dx = 2
dy = 2

nx = 4
ny = 4

N = nx*ny

radius = 0.5

time = 15000
Temp = 0.5
T_ext = 0.5
k = 1 



m = 1

dt = 0.001

boxsize = 8
promien = 0.5
V=boxsize**2
#%% 

# funkcje 

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
        particles.append(czastka(radius, pol, vel,np.empty((time,2)),np.empty((time,2))))


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
        f = (48*eps/sig)*((sig/r12)**13 - (1/2)*(sig/r12)**7)  
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
    


# inicjalizacja

F1 = []
for pi in particles:
    f1=0
    for pj in particles:
        if not pi==pj:
            f1+=F(pi.r,pj.r)
    F1.append(f1)
    
# cofamy prędkość o półkroku    
for pi in particles:
    pi.v = pi.v - (F1[particles.index(pi)]*dt)/m        

#%%

T,P=np.empty(time),np.empty(time)
E,E_kin,E_pot=np.empty(time),np.empty(time),np.empty(time)

for i in range(time):
    if (i%500==0):
        print(i)
        
    K1=0
    fr=0
    U=0
    
    '''liczenie siły, energii potencjalnej, iloczynów r*F dla ciśnienia'''
    F1=np.empty((N,2))
    for pi in particles:
        
        ind = particles.index(pi)   
        f1=0
        
        for pj in particles:
            if not pi==pj:
               f=F(pi.r,pj.r)
               f1 += f
               
               r12 = closest_image(pi.r,pj.r)#pi.r-pj.r##
               fr += 0.5*np.dot(r12,f)
               U += 0.5*potencjal(pi.r,pj.r)
        
        
        '''przesuwamy prędkość o półkroku bez tłumienia - "krok wirtualny"'''
        v01 = pi.v
        v1 = v01 + (f1*dt)/(2*m)
        
        '''energia kinetyczna'''
        K1+= (m/2)*(v1[0]**2+v1[1]**2)
        
        F1[ind] = f1
        
        
    t1 = K1/N
    ''' eta ! '''
    eta = np.sqrt(T_ext/t1) 
    
    

    K=0
    for pi in particles:
        ind = particles.index(pi)   
        
        '''aktualizujemy prędkości i położenia'''
        
        r1 = pi.r
        v01 = pi.v
        
        v12 = (2*eta-1)*v01 + (eta*F1[ind]*dt)/m
        r2 = bound(r1 + v12*dt)
        
        ''' v1 otrzymujemy z obecnego i poprzedniego kroku'''
        v1 = (v01+v12)/2
        
        ''' "prawdziwa" energia kinetyczna'''
        K+=(m/2)*(v1[0]**2+v1[1]**2)
        
        pi.r = r2
        pi.v = v12
        
        pi.his[i,:] = pi.r
        pi.hisv[i,:] = pi.v
        
    
    ''' temperatura, ciśnienie, energie'''    
    t=K/N
    T[i]=t
    
    p = (N*t)/V - fr/(2*V)
    P[i]=p
    
    
    E_kin[i]=K
    E_pot[i]=U
    E[i]=K+U
    
    # animacja
    if (i%50==0):
        plt.clf()
        fig= plt.gcf()
        for p in particles:
            a = plt.gca()
            cir= Circle((p.r[0],p.r[1]), radius=p.promien) 
            a.add_patch(cir)
        plt.plot()
        plt.title('Krok '+str(i))
        plt.xlim((0,boxsize))
        plt.ylim((0,boxsize))    
        fig.set_size_inches((6,6))
        plt.savefig('/home/dell/sym_komp/4-termostat/25_cz_T=0.7/img'+str(i)+'.png') 
        plt.show()


#%% rysowanie 1
czas = np.arange(time)*dt
plt.rcParams['figure.figsize'] = (20,10) 
plt.plot(czas,T,label='T',color='blue')
plt.plot(czas,P,label='P',color='darkorange')
plt.plot(czas,E/N,label='E/N',color='forestgreen')
#plt.title('Dochodzenie do równowagi')
plt.xlabel('Czas [s]',fontsize=15)
plt.legend(fontsize=15)
#plt.savefig('/home/dell/sym_komp/4-termostat/25_cz_T=0.7/25_cz_E,P,T=1.3_dt=0.001.png')
plt.show()

#%% rysowanie 2

plt.plot(czas,E_kin,label='K',color='indigo')
plt.plot(czas,E_pot,label='U',color='tomato')
plt.plot(czas,E,label='E',color='forestgreen')
#plt.title('Dochodzenie do równowagi')
plt.xlabel('Czas [s]',fontsize=15)
plt.legend(fontsize=15)
#plt.savefig('/home/dell/sym_komp/4-termostat/25_cz_T=0.7/25_cz_E,K,U_T=1.3_dt=0.001.png')
plt.show()
#%% filmiki

fps=20

image_files = []
for i in range(299):
    image_files.append('/home/dell/sym_komp/4-termostat/T=1.3/'+'img'+str(i*50)+'.png')    


clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('T=1.3_dt=0.001.mp4')