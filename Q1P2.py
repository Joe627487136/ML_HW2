import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from numpy.linalg import inv
from scipy.optimize import fmin_l_bfgs_b as minimize

ifile = open("linear.csv","rb")
data=np.genfromtxt(ifile,delimiter=",")
hX=data[:,1:4] #all X
hY=data[:,0]  #all Y
vX=data[0:10,1:4]
vY=data[0:10,0]
tX=data[10:50,1:4]
tY=data[10:50,0]
learn_rate=0.5  #learning rate for gradient descent
d=vX.shape[1]   #columns num
hn=hX.shape[0]    #all data amount
vn=vX.shape[0]    #validation data amount
tn=tX.shape[0]    #training data amount
x=T.matrix(name="x")  #feature matrix
y=T.vector(name="y")  #response vector
w=theano.shared(np.zeros((d,1)),name="w")  #model parameters
b=theano.shared(0., name='b') #not penalized w0 parameters
model = T.dot(x,w).T+b  #seperate W0
e_risk=T.sum((model-y)**2)/2.0/tn  #empiricalrisk
l=0.15 #lamda
risk=e_risk+l*0.5*(T.sum(w**2))  #Featured with ridge
gw=T.grad(risk,wrt=w)   #gradientoftherisk for w
gb=T.grad(risk,wrt=b)   #gradientoftherisk for w0
updates = ((w,w-learn_rate*gw), (b,b-learn_rate*gb))
train_model=theano.function(inputs=[],outputs=risk,updates=updates,givens={x:tX,y:tY})
n_steps=50
for i in range(n_steps):
    print(train_model())
print(w.get_value())
print(b.get_value())
ifile.close()



