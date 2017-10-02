import numpy as np
from numpy.linalg import inv

ifile = open("linear.csv","rb")
data=np.genfromtxt(ifile,delimiter=",")
hX=data[:,1:4] #all X
hY=data[:,0]  #all Y
vX=data[0:10,1:]
vY=data[0:10,0]
tX=data[10:50,1:]
tY=data[10:50,0]

def ridge_regression(tX, tY, l):
    n = tX.shape[0]
    J = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], np.int32)
    xty = np.dot(np.transpose(tX), tY)
    xtx = np.dot(np.transpose(tX), tX)
    w = np.dot(inv(n * l * J + xtx), xty)
    return w

print(ridge_regression(tX,tY,0.15))