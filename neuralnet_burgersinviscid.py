### Neural network to solve inviscid Burgers equation

import autograd.numpy as np
from autograd import elementwise_grad as grad
from scipy.optimize import minimize
import time

def f(u,ut,ux): 
    return ut+u*ux

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

def net(x,t,a,p,dist,layers):
    y = np.stack([x,t,a])
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

def loss(p,x,t,a,dist,layers):
    lossf = np.sum(f(net(x,t,a,p,dist,layers),ut(x,t,a,p,dist,layers),ux(x,t,a,p,dist,layers))**2)/len(ttrain)
    losscc = np.sum((acc*xcc-net(xcc,np.zeros(len(xcc)),acc,p,dist,layers))**2)/len(xcc)
    return lossf+losscc
#%% Entrenamiento de la red
# Dominio de entrenamiento
xcc = np.linspace(-1,1,100)
tcc = np.linspace(0,1,100)
acc = np.linspace(1,5,5)
meshcc = np.meshgrid(xcc,acc)
meshtrain = np.meshgrid(xcc,tcc,acc)
xtrain = np.reshape(meshtrain[0],50000)
ttrain = np.reshape(meshtrain[1],50000)
atrain = np.reshape(meshtrain[2],50000)
xcc = np.reshape(meshcc[0],500)
acc = np.reshape(meshcc[1],500)


# Parámetros iniciales
layers = [3,20,20,10,1]
dist = distribution(layers)
p = xavier(layers)
# Entrenamiento
jacloss = grad(loss)
t0 = time.time()
res = minimize(loss,p,args=(xtrain,ttrain,atrain,dist,layers),method='BFGS',jac=jacloss,options={'maxiter':1000,'disp':True})
ttraining = time.time()-t0
p = res.x
print('Loss: %.7f ' % res.fun)
print('Tiempo de ejecucion: %.2f segundos' % ttraining)
#%% Comprobación de la solución
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

meshcomp = np.meshgrid(np.linspace(-1,1,200),np.linspace(0,1,200))
acomp = 4.5

def exact(x,t):
    return acomp*x/(1+acomp*t)

uex = exact(meshcomp[0],meshcomp[1])

p = np.load('burgersinviscidparametrodata.npy')

fig1 = plt.figure()
plt.contourf(meshcomp[1],meshcomp[0],uex)
plt.title('Solución exacta')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.show()

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(meshcomp[0],meshcomp[1],uex,rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)

ax2.set_xlim(-1,1)
ax2.set_ylim(0,1)
ax2.set_title('Solución exacta')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$t$');

unet = np.reshape(net(np.reshape(meshcomp[0],40000),np.reshape(meshcomp[1],40000),acomp*np.ones(40000),p,dist,layers),(200,200))

fig3 = plt.figure()
plt.contourf(meshcomp[1],meshcomp[0],unet)
plt.title('Solución de la red')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.show()

fig4 = plt.figure()
ax4 = fig4.gca(projection='3d')
surf4 = ax4.plot_surface(meshcomp[0],meshcomp[1],unet,rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)

ax4.set_xlim(0,1)
ax4.set_ylim(-1,1)
ax4.set_title('Solución de la red')
ax4.set_xlabel('$t$')
ax4.set_ylabel('$x$');

error = np.abs(uex-unet)

fig5 = plt.figure()
plt.contourf(meshcomp[1],meshcomp[0],error)
plt.title('Error')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.show()

fig6 = plt.figure()
ax6 = fig6.gca(projection='3d')
surf6 = ax6.plot_surface(meshcomp[0],meshcomp[1],error,rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)

ax6.set_xlim(0,1)
ax6.set_ylim(-1,1)
ax6.set_title('Error')
ax6.set_xlabel('$t$')
ax6.set_ylabel('$x$');

