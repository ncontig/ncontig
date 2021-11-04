import autograd.numpy as np
from autograd import elementwise_grad as grad
from scipy.optimize import minimize
import time

alfa = 0.01

def f(Tt,Txx): # Tt = alfa*Txx
    return Tt-alfa*Txx

def f1(x): # C.I.
    return np.sin(np.pi*x)

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

def net(t,x,p,dist,layers):
    y = np.stack([t,x])
    for i in range(len(layers)-2):    
        w = np.reshape(p[dist[2*i]:dist[2*i+1]],(layers[i+1],layers[i]))
        b = np.reshape(p[dist[2*i+1]:dist[2*i+2]],(layers[i+1],1))
        y = np.tanh(np.dot(w,y)+b)
    w = np.reshape(p[dist[-3]:dist[-2]],(layers[-1],layers[-2])) 
    b = p[-1]
    y = np.sum((np.dot(w,y)+b),axis=0)
    return y

Tt = grad(net)   
Txx = grad(grad(net,argnum=1),argnum=1)

def loss(p,ttrain,xtrain,tcc,xcc,dist,layers):
    lossf = np.sum(f(Tt(ttrain,xtrain,p,dist,layers),Txx(ttrain,xtrain,p,dist,layers))*f(Tt(ttrain,xtrain,p,dist,layers),Txx(ttrain,xtrain,p,dist,layers)))/len(ttrain)
    lossci = np.sum((net(np.zeros(len(xcc)),xcc,p,dist,layers)-f1(xcc))*(net(np.zeros(len(xcc)),xcc,p,dist,layers)-f1(xcc)))/len(xcc)
    losscc1 = np.sum(net(tcc,np.zeros(len(tcc)),p,dist,layers)*net(tcc,np.zeros(len(tcc)),p,dist,layers))/len(tcc) 
    losscc2 = np.sum(net(tcc,np.ones(len(tcc)),p,dist,layers)*net(tcc,np.ones(len(tcc)),p,dist,layers))/len(tcc) 
    return lossf+lossci+losscc1+losscc2
#%% Entrenamiento de la red
# Condiciones de contorno
xcc = np.linspace(0,1,10)
tcc = np.linspace(0,10,10)

# Dominio de entrenamiento
ttrain = np.linspace(0,10,10)
xtrain = np.linspace(0,1,10)
ttrain,xtrain = np.meshgrid(ttrain,xtrain)
ttrain = np.reshape(ttrain,100)
xtrain = np.reshape(xtrain,100)

# Parámetros iniciales
layers = [2,20,1]
dist = distribution(layers)
p0 = xavier(layers)

# Entrenamiento
jacloss = grad(loss)
t0 = time.time()
res = minimize(loss,p0,args=(ttrain,xtrain,tcc,xcc,dist,layers),method='BFGS',jac=jacloss,options={'maxiter':10000,'disp':True})
ttraining = time.time()-t0
p = res.x
print('Loss: %.7f ' % res.fun)
print('Tiempo de ejecucion: %.2f segundos' % ttraining)
#%% Comprobación de la solución
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def exact(t,x): # Solución exacta de la EDP
    return np.sin(np.pi*x)*np.exp(-alfa*np.pi**2*t)

tcomp = np.linspace(0,10,1000)
xcomp = np.linspace(0,1,100)
ttcomp,xxcomp = np.meshgrid(tcomp,xcomp)

fig1 = plt.figure()
phiexact = exact(ttcomp,xxcomp)
plt.contourf(ttcomp,xxcomp,phiexact)
plt.title('Solución exacta')
plt.colorbar()
plt.show()

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(ttcomp,xxcomp,phiexact,rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)

ax2.set_xlim(0,10)
ax2.set_ylim(0,1)
ax2.set_title('Solución exacta')
ax2.set_xlabel('$t$')
ax2.set_ylabel('$x$');

fig3 = plt.figure()
phinet = np.reshape(net(np.reshape(ttcomp,(100000)),np.reshape(xxcomp,(100000)),p,dist,layers),(100,1000))
plt.contourf(ttcomp,xxcomp,phinet)
plt.title('Solución de la red')
plt.colorbar()
plt.show()

fig4 = plt.figure()
ax4 = fig4.gca(projection='3d')
surf4 = ax4.plot_surface(ttcomp,xxcomp,phinet,rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)

ax4.set_xlim(0,10)
ax4.set_ylim(0,1)
ax4.set_title('Solución de la red')
ax4.set_xlabel('$t$')
ax4.set_ylabel('$x$');

fig5 = plt.figure()
error = np.abs(phiexact-phinet)
plt.contourf(ttcomp,xxcomp,error)
plt.title('Error')
plt.colorbar()
plt.show()

fig6 = plt.figure()
ax6 = fig6.gca(projection='3d')
surf6 = ax6.plot_surface(ttcomp,xxcomp,error,rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)

ax6.set_xlim(0,10)
ax6.set_ylim(0,1)
ax6.set_title('Error')
ax6.set_xlabel('$t$')
ax6.set_ylabel('$x$');


