import autograd.numpy as np
from autograd import elementwise_grad as grad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time

def f(Tt,Txx): # dT/dt = d2T/dx2
    return Tt-Txx

def f1(x): # CC en t = 0
    return np.sin(np.pi*x)

def g1(x): # CC en y = 0
    return x**2

def g2(x): # CC en y = 2
    return 8+x**2

def sigmoid(x): # Función de activación
    return 1/(1+np.exp(-x))

def net(x,y,p):
    w = p[0:30]
    b = p[30:41]
    y1 = np.array([x,y])
    w12 = np.reshape(w[0:20],(10,2))
    b1 = np.reshape(b[0:10],(10,1))
    y2 = sigmoid(np.dot(w12,y1)+b1)
    w23 = np.reshape(w[20:30],(1,10)) 
    b2 = b[10]
    y3 = np.sum((np.dot(w23,y2)+b2),axis=0)
    return y3

def nety(y,x,p):
    w = p[0:30]
    b = p[30:41]
    y1 = np.array([x,y])
    w12 = np.reshape(w[0:20],(10,2))
    b1 = np.reshape(b[0:10],(10,1))
    y2 = sigmoid(np.dot(w12,y1)+b1)
    w23 = np.reshape(w[20:30],(1,10)) 
    b2 = b[10]
    y3 = np.sum((np.dot(w23,y2)+b2),axis=0)
    return y3

grad2xnet = grad(grad(net))
grad2ynet = grad(grad(nety))

def loss(p,x,y):
    loss_eq = np.sum((grad2xnet(x,y,p)+grad2ynet(y,x,p)-f(x,y))**2)
    v1 = np.linspace(0,0,5)
    v2 = np.linspace(2,2,5)
    v3 = np.linspace(0,2,5)   
    loss_cc_x1 = np.sum((net(v1,v3,p)-f1(v3))**2)
    loss_cc_x2 = np.sum((net(v2,v3,p)-f2(v3))**2)
    loss_cc_y1 = np.sum((net(v3,v1,p)-g1(v3))**2)
    loss_cc_y2 = np.sum((net(v3,v2,p)-g2(v3))**2)
    return loss_eq+loss_cc_x1+loss_cc_x2+loss_cc_y1+loss_cc_y2

gradloss = grad(loss)

x = np.linspace(0,2,5) # Dominio de entrenamiento
y = np.linspace(0,2,5)
x,y = np.meshgrid(x,y) # Dominio de entrenamiento
x = np.reshape(x,(25)) 
y = np.reshape(y,(25))

#%% Entrenamiento de la red
p0 = np.random.rand(41) # Parámetros iniciales
#p0 = np.load('poisson_p0.npy')
t0 = time.time()
res = minimize(loss,p0,args=(x,y)) # Entrenamiento de la red
ttrain = time.time()-t0
p = res.x # Parámetros optimizados

#%% Comprobación de la solución
def exact(x,y): # Solución exacta de la EDO
    return x**2+y**3

xcomp = np.linspace(0,2,20)
ycomp = np.linspace(0,2,20)
xxcomp,yycomp = np.meshgrid(xcomp,ycomp)

fig1 = plt.figure()
phiexact = exact(xxcomp,yycomp)
plt.contourf(xxcomp,yycomp,phiexact)
plt.title('Solución exacta')
plt.colorbar()
plt.show()

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(xxcomp,yycomp,phiexact,rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)

ax2.set_xlim(0,2)
ax2.set_ylim(0,2)
ax2.set_title('Solución exacta')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$');

fig3 = plt.figure()
phinet = np.reshape(net(np.reshape(xxcomp,(400)),np.reshape(yycomp,(400)),p),(20,20))
plt.contourf(xxcomp,yycomp,phinet)
plt.title('Solución de la red')
plt.colorbar()
plt.show()

fig4 = plt.figure()
ax4 = fig4.gca(projection='3d')
surf4 = ax4.plot_surface(xxcomp,yycomp,phinet,rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)

ax4.set_xlim(0,2)
ax4.set_ylim(0,2)
ax4.set_title('Solución de la red')
ax4.set_xlabel('$x$')
ax4.set_ylabel('$y$');

fig5 = plt.figure()
error = np.abs(phiexact-phinet)
plt.contourf(xxcomp,yycomp,error)
plt.title('Error')
plt.colorbar()
plt.show()

fig6 = plt.figure()
ax6 = fig6.gca(projection='3d')
surf6 = ax6.plot_surface(xxcomp,yycomp,error,rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)

ax6.set_xlim(0,2)
ax6.set_ylim(0,2)
ax6.set_title('Error')
ax6.set_xlabel('$x$')
ax6.set_ylabel('$y$');


