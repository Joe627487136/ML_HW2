import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from numpy.linalg import inv
from scipy.optimize import fmin_l_bfgs_b as minimize

ifile = open("linear.csv","rb")
data=np.genfromtxt(ifile,delimiter=",")
hX=data[:,1:] #all X
hY=data[:,0]  #all Y
vX=data[0:10,1:]
vY=data[0:10,0]
tX=data[10:50,1:]
tY=data[10:50,0]
tn=tX.shape[0]    #training data amount

myY=np.array(tY)


def costgrad(w, x, y, l):
    cost = np.sum((np.dot(x,w) - y) ** 2)/2/tn + l/2*np.sum(np.square(w[:-1]))
    grad = l*np.append(w[:-1],0.) + 1/tn*(np.dot(np.dot(np.transpose(x), x),w)) - 1/tn * np.dot(np.transpose(x),y)
    return cost,grad

w = np.random.rand(4, 1)
print(w)
l=0.15
optx,cost,messages=minimize(costgrad,x0=w, args=[tX,tY,l])
print(optx)