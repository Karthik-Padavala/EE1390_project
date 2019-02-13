import numpy as np
import matplotlib.pyplot as plt

def dir_vec(AB):
 return np.matmul(AB,dvec)

def norm_vec(AB):
 return np.matmul(omat,np.matmul(AB,dvec))

def line_intersect(XY):
 n1= dir_vec(XY)
 n2= norm_vec(XY)
 N = np.vstack((n1,n2))
 p = np.zeros(2)
 p[0] = np.matmul(n1,[0,0])
 p[1] = np.matmul(n2,XY[:,0])
 return np.matmul(np.linalg.inv(N),p)

X = np.array([2,0])
Y = np.array([0,2])

U = np.array([0,-4])
V = np.array([-4,0])

dvec = np.array([-1,1])
omat = np.array([[0,1],[-1,0]])
XY = np.vstack((X,Y)).T
UV = np.vstack((U,V)).T

E = dir_vec(XY)
D = norm_vec(XY)
P = line_intersect(XY)
Q = line_intersect(UV)
Z = np.linalg.norm(P-Q)

A = (np.sqrt(3)*Z)/2

area = (A*Z)/2

print(area)
print(A)
print(Z)
print(Q)
print(P)
print(E)
print(D)

len=10

lam_1 = np.linspace(0,1,len)

x_XY = np.zeros((2,len))
x_UV = np.zeros((2,len))
x_PQ = np.zeros((2,len))

for i in range(len):
 temp1 = X +lam_1[i]*(Y-X)
 x_XY[:,i]= temp1.T
 temp2 = U +lam_1[i]*(V-U)
 x_UV[:,i]= temp2.T
 temp3 = P +lam_1[i]*(Q-P)
 x_PQ[:,i]= temp3.T

plt.plot(x_XY[0,:],x_XY[1,:],label='$XY$')
plt.plot(x_UV[0,:],x_UV[1,:],label='$UV$')
plt.plot(x_PQ[0,:],x_PQ[1,:],label='$PQ$')


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()

plt.show()







