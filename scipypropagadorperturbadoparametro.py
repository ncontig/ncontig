import autograd.numpy as np
from autograd import elementwise_grad as grad
from scipy.optimize import minimize
import time

def f(ap,x,y,z,xt,yt,zt,xtt,ytt,ztt): # f(y,uxx,uyy) = 0
    f1 = xtt+x/(x**2+y**2+z**2)**1.5-ap*xt/np.sqrt(xt**2+yt**2+zt**2)
    f2 = ytt+y/(x**2+y**2+z**2)**1.5-ap*yt/np.sqrt(xt**2+yt**2+zt**2)
    f3 = ztt+z/(x**2+y**2+z**2)**1.5-ap*zt/np.sqrt(xt**2+yt**2+zt**2)
    return np.array([f1,f2,f3])

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

def net(t,ap,p,dist,layers):
    y = np.stack([t,ap])
    for i in range(len(layers)-2):    
        w = np.reshape(p[dist[2*i]:dist[2*i+1]],(layers[i+1],layers[i]))
        b = np.reshape(p[dist[2*i+1]:dist[2*i+2]],(layers[i+1],1))
        y = np.tanh(np.dot(w,y)+b)
    w = np.reshape(p[dist[-3]:dist[-2]],(layers[-1],layers[-2])) 
    b = p[-1]
    y = np.sum((np.dot(w,y)+b),axis=0)
    return y

def trialx(t,ap,p,dist,layers):
    return x0+vx0*t+t**2*net(t,ap,p,dist,layers)

def trialy(t,ap,p,dist,layers):
    return y0+vy0*t+t**2*net(t,ap,p,dist,layers)

def trialz(t,ap,p,dist,layers):
    return z0+vz0*t+t**2*net(t,ap,p,dist,layers)

xt = grad(trialx)
yt = grad(trialy) 
zt = grad(trialz)  
xtt = grad(grad(trialx))
ytt = grad(grad(trialy))
ztt = grad(grad(trialz))   

def loss(p,ttrain,aptrain,dist,layers):
    px = p[0:dist[-1]]
    py = p[dist[-1]:dist[-1]+dist[-1]]
    pz = p[dist[-1]+dist[-1]:dist[-1]+dist[-1]+dist[-1]]
    loss = np.sum(f(aptrain,trialx(ttrain,aptrain,px,dist,layers),trialy(ttrain,aptrain,py,dist,layers),trialz(ttrain,aptrain,pz,dist,layers),xt(ttrain,aptrain,px,dist,layers),yt(ttrain,aptrain,py,dist,layers),zt(ttrain,aptrain,pz,dist,layers),xtt(ttrain,aptrain,px,dist,layers),ytt(ttrain,aptrain,py,dist,layers),ztt(ttrain,aptrain,pz,dist,layers))*f(aptrain,trialx(ttrain,aptrain,px,dist,layers),trialy(ttrain,aptrain,py,dist,layers),trialz(ttrain,aptrain,pz,dist,layers),xt(ttrain,aptrain,px,dist,layers),yt(ttrain,aptrain,py,dist,layers),zt(ttrain,aptrain,pz,dist,layers),xtt(ttrain,aptrain,px,dist,layers),ytt(ttrain,aptrain,py,dist,layers),ztt(ttrain,aptrain,pz,dist,layers)))/len(ttrain)
    return loss
#%% Entrenamiento de la red
# Condiciones de contorno
mu = 398600.
Re = 6378.
w = np.sqrt(mu/Re**3)
xdim0 = 8000.
x0 = xdim0/Re
vxdim0 = 0.
vx0 = vxdim0/(Re*w)
ydim0 = 0.
y0 = ydim0/Re
vydim0 = np.sqrt(mu/xdim0)/np.sqrt(2)
vy0 = vydim0/(Re*w)
zdim0 = 0.
z0 = zdim0/Re
vzdim0 = vydim0
vz0 = vzdim0/(Re*w)

r0 = np.array([xdim0,ydim0,zdim0])
v0 = np.array([vxdim0,vydim0,vzdim0])

# Parámetros keplerianos
a = np.linalg.norm(r0)/(2-np.linalg.norm(r0)*np.linalg.norm(v0)**2/mu)
T = 2*np.pi*np.sqrt(a**3/mu)
e = np.linalg.norm(-r0/np.linalg.norm(r0)-np.cross(np.cross(r0,v0),v0)/mu)
i = np.arcsin(vzdim0/np.linalg.norm(v0))

# Dominio de entrenamiento
ttrain = np.linspace(0,3*T*w,1000)
aptrain = np.linspace(0.01,0.1,10)
tt,aa = np.meshgrid(ttrain,aptrain)
ttrain = np.reshape(tt,10000)
aptrain = np.reshape(aa,10000)

# Parámetros iniciales
layers = [2,20,10,1]
dist = distribution(layers)
px0 = xavier(layers)
py0 = xavier(layers)
pz0 = xavier(layers)
p0 = np.concatenate((px0,py0,pz0))
# Entrenamiento
jacloss = grad(loss)
t0 = time.time()
res = minimize(loss,p0,args=(ttrain,aptrain,dist,layers),method='BFGS',jac=jacloss,options={'maxiter':10000,'disp':True})
ttraining = time.time()-t0
p = res.x
px = p[0:dist[-1]]
py = p[dist[-1]:dist[-1]+dist[-1]]
pz = p[dist[-1]+dist[-1]:dist[-1]+dist[-1]+dist[-1]]
print('Loss: %.7f ' % res.fun)
print('Tiempo de ejecucion: %.2f segundos' % ttrain)
#%% Comprobación
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import newton

tcomp = np.linspace(0,3*T*w,5000)
acomp = np.linspace(0.035,0.035,5000)
xnet = trialx(tcomp,acomp,px,dist,layers)*Re
ynet = trialy(tcomp,acomp,py,dist,layers)*Re
znet = trialz(tcomp,acomp,pz,dist,layers)*Re
xtnet = xt(tcomp,acomp,px,dist,layers)
ytnet = yt(tcomp,acomp,py,dist,layers)
ztnet = zt(tcomp,acomp,pz,dist,layers)
xttnet = xtt(tcomp,acomp,px,dist,layers)
yttnet = ytt(tcomp,acomp,py,dist,layers)
zttnet = ztt(tcomp,acomp,pz,dist,layers)
zzz = f(acomp,xnet/Re,ynet/Re,znet/Re,xtnet,ytnet,ztnet,xttnet,yttnet,zttnet)
losstest = np.sum(zzz*zzz)/len(zzz)

fig1 = plt.figure()
plt.plot(xnet,ynet)
plt.axis('equal')
plt.title('Solución de la red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig2 = plt.figure()
ax = fig2.gca(projection='3d')
ax.plot(xnet,ynet,znet)
plt.title('Solución de la red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

Mcomp = np.sqrt(mu/a**3)*tcomp/w
def kepler(M,e):
    def kepler2(E,M,e):
        return(M-(E-e*np.sin(E)))
    E0 = M
    E = newton(kepler2,E0,args=(M,e))
    return E
Ecomp = kepler(Mcomp,e)
def exact(E):
    r = a*(1-e*np.cos(E))
    theta = 2*np.arctan2(np.sqrt((1+e)/(1-e))*np.sin(E/2),np.cos(E/2))
    return r*np.cos(theta),r*np.cos(i)*np.sin(theta),r*np.sin(i)*np.sin(theta)

xexact,yexact,zexact = exact(Ecomp)

fig3 = plt.figure()
plt.plot(xexact,yexact)
plt.axis('equal')
plt.title('Solución exacta')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig4 = plt.figure()
ax = fig4.gca(projection='3d')
ax.plot(xexact,yexact,zexact)
plt.title('Solución exacta')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

errorx = np.minimum(np.abs((xexact-xnet)/(xexact+1)),0.1)
errory = np.minimum(np.abs((yexact-ynet)/(yexact+1)),0.1)
errorz = np.minimum(np.abs((zexact-znet)/(zexact+1)),0.1)

plt.plot(tcomp/w,errorx)
plt.title('Error según x')
plt.xlabel('t')
plt.ylim((0,0.5))
plt.show()

plt.plot(tcomp/w,errory)
plt.title('Error según y')
plt.xlabel('t')
plt.ylim((0,0.5))
plt.show()

plt.plot(tcomp/w,errorz)
plt.title('Error según z')
plt.xlabel('t')
plt.ylim((0,0.5))
plt.show()
