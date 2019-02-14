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

# Let A,B,C be the vertices of the triangle . Centroid is origin and two points are on the line [ (1 1) X = 2 ]  i.e  [ (1 1) A = 2] and [ (1 1) B = 2]  and [ (1 1) C = -4 ]
# third vertex passes through the line [ (1 1) C = -4 ]

X = np.array([2,0])       # the point where the given line touches X-axis
Y = np.array([0,2])	  # the point where the given line touches Y-axis   [ (1 1) X = 2 ]
			 
U = np.array([0,-4])	  # the points where the line [ (1 1) C = -4 ] touches axes
V = np.array([-4,0])

dvec = np.array([-1,1])
omat = np.array([[0,1],[-1,0]])
XY = np.vstack((X,Y)).T
UV = np.vstack((U,V)).T

N = np.array([1,-1])

P = line_intersect(XY)     # P is point of intersection of line passing through origin and perpendicular to the line [ (1 1) X = 2 ] and [ (1 1) X = 2 ]
Q = line_intersect(UV)     # P is point of intersection of line passing through origin and perpendicular to the line [ (1 1) C = -4 ] and [ (1 1) C = -4 ]

# The line passing through P and Q is the heightof the triangle
 
H = np.linalg.norm(P-Q)    # Height of the triangle

L = (2*H)/np.sqrt(3)	   # Length of the side of the triangle

area = (L*H)/2		   # Area of the triangle



M = L/(2*np.sqrt(2))

A = P+(M*N)		  # One vertex of the triangle which is on the line [ (1 1) X = 2]
B = P-(M*N)
G = A+B+Q		  # One vertex of the triangle which is on the line [ (1 1) X = 2]

print(A)
print(B)
print(Q)
print(H)
print(L)
print(area)

len=10

lam_1 = np.linspace(0,1,len)

x_XY = np.zeros((2,len))
x_UV = np.zeros((2,len))
x_PQ = np.zeros((2,len))
x_BQ = np.zeros((2,len))
x_AQ = np.zeros((2,len))
x_AB = np.zeros((2,len))

for i in range(len):
 temp1 = X +lam_1[i]*(Y-X)
 x_XY[:,i]= temp1.T
 temp2 = U +lam_1[i]*(V-U)
 x_UV[:,i]= temp2.T
 temp3 = P +lam_1[i]*(Q-P)
 x_PQ[:,i]= temp3.T
 temp4 = B +lam_1[i]*(Q-B)
 x_BQ[:,i]= temp4.T
 temp5 = A +lam_1[i]*(Q-A)
 x_AQ[:,i]= temp5.T
 temp6 = A +lam_1[i]*(B-A)
 x_AB[:,i]= temp6.T



plt.plot(x_XY[0,:],x_XY[1,:],label='$XY$')
plt.plot(x_UV[0,:],x_UV[1,:],label='$UV$')
plt.plot(x_PQ[0,:],x_PQ[1,:],label='$PQ$')
plt.plot(x_BQ[0,:],x_BQ[1,:],label='$BQ$')
plt.plot(x_AQ[0,:],x_AQ[1,:],label='$AQ$')
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')

plt.plot(U[0],U[1],'o')
plt.text(U[0]*(1+0.1),U[1]*(1),'U')
plt.plot(V[0],V[1],'o')
plt.text(V[0]*(1+0.1),V[1]*(1),'V')
plt.plot(X[0],X[1],'o')
plt.text(X[0]*(1+0.1),X[1]*(1),'X')
plt.plot(Y[0],Y[1],'o')
plt.text(Y[0]*(1+0.1),Y[1]*(1-0.1),'Y')
plt.plot(P[0],P[1],'o')
plt.text(P[0]*(1+0.1),P[1]*(1),'P')
plt.plot(Q[0],Q[1],'o')
plt.text(Q[0]*(1+0.1),Q[1]*(1),'Q')
plt.plot(A[0],A[1],'o')
plt.text(A[0]*(1+0.1),A[1]*(1),'A')
plt.plot(B[0],B[1],'o')
plt.text(B[0]*(1+0.1),B[1]*(1),'B')
plt.plot(G[0],G[1],'o')
plt.text(G[0]*(1+0.1),G[1]*(1),'G')


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()

plt.show()







