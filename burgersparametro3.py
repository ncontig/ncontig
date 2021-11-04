import autograd.numpy as np
from autograd import elementwise_grad as grad
from scipy.optimize import minimize
import time

def f(v,u,ut,ux,uxx): 
    return ut+u*ux-v*uxx

def distribution(layers):
    dist = [0]
    for i in range(len(layers)-1):
        dist.append(dist[2*i]+layers[i]*layers[i+1])
        dist.append(dist[2*i+1]+layers[i+1])
    return dist

def xavier(layers):
    p = np.array([])
    for i in range(len(layers)-2):
        w = np.sqrt(6/(layers[i]*layers[i+1]+layers[i+1]*layers[i+2]))*(-1+2*np.random.rand(layers[i]*layers[i+1]))
        p = np.append(p,w)
        b = np.zeros(layers[i+1])
        p = np.append(p,b)
    w = np.sqrt(6/(layers[-2]*layers[-1]))*(-1+2*np.random.rand(layers[-2]*layers[-1]))
    p = np.append(p,w)
    b = np.zeros(layers[-1])
    p = np.append(p,b)
    return p

def net(x,t,v,p,dist,layers):
    y = np.stack([x,t,v])
    for i in range(len(layers)-2):    
        w = np.reshape(p[dist[2*i]:dist[2*i+1]],(layers[i+1],layers[i]))
        b = np.reshape(p[dist[2*i+1]:dist[2*i+2]],(layers[i+1],1))
        y = np.tanh(np.dot(w,y)+b)
    w = np.reshape(p[dist[-3]:dist[-2]],(layers[-1],layers[-2])) 
    b = p[-1]
    y = np.sum((np.dot(w,y)+b),axis=0)
    return y

ut = grad(net,argnum=1)
ux = grad(net,argnum=0)
uxx = grad(ux,argnum=0) 

def loss(p,x,t,v,dist,layers):
    lossf = np.sum(f(v,net(x,t,v,p,dist,layers),ut(x,t,v,p,dist,layers),ux(x,t,v,p,dist,layers),uxx(x,t,v,p,dist,layers))**2)/len(t)
    lossccx = np.sum((-np.sin(np.pi*xcc)-net(xcc,np.zeros(len(xcc)),vcc,p,dist,layers))**2)/len(xcc)
    losscct = (np.sum(net(-np.ones(len(tcc)),tcc,vcc,p,dist,layers)**2)+np.sum(net(np.ones(len(tcc)),tcc,vcc,p,dist,layers)**2))/(2*len(tcc))
    return lossf+lossccx+losscct

jacloss = grad(loss)
#%% Entrenamiento de la red
# Dominio de entrenamiento
xcc = np.linspace(-1,1,100)
tcc = np.linspace(0,1,100)
vcc = np.linspace(0.01,0.1,5)
meshtrain = np.meshgrid(xcc,tcc,vcc)
meshccx = np.meshgrid(xcc,vcc)
meshcct = np.meshgrid(tcc,vcc)
xtrain = np.reshape(meshtrain[0],50000)
ttrain = np.reshape(meshtrain[1],50000)
vtrain = np.reshape(meshtrain[2],50000)
xcc = np.reshape(meshccx[0],500)
tcc = np.reshape(meshcct[0],500)
vcc = np.reshape(meshccx[1],500)

# Parámetros iniciales
layers = [3,20,20,20,20,10,1]
dist = distribution(layers)
p = xavier(layers)

p = np.load('burgersparametrodata3.npy')

#%% Comprobación de la solución
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

xcomp = np.linspace(-1,1,100)
tcomp = np.linspace(0,1,100)
meshcomp = np.meshgrid(xcomp,tcomp)
vcomp = 0.015*np.ones(10000)

unet = np.reshape(net(np.reshape(meshcomp[0],10000),np.reshape(meshcomp[1],10000),vcomp,p,dist,layers),(100,100))

fig1 = plt.figure()
plt.contourf(meshcomp[1],meshcomp[0],unet)
plt.title('Solución de la red')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.show()

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(meshcomp[0],meshcomp[1],unet,rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)

ax2.set_xlim(-1,1)
ax2.set_ylim(0,1)
ax2.view_init(azim=60)
ax2.set_title('Solución de la red')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$t$');
