import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import csv

ifile = open("linear.csv","rb")
reader = csv.reader(ifile)
csv="https://www.dropbox.com/s/oqoyy9p849ewzt2/linear.csv?dl=1"
data=np.genfromtxt(ifile,delimiter=",")
X=data[:,1:]
Y=data[:,0]
d=X.shape[1]    #columns num
n=X.shape[0]    #data amount
learn_rate=0.5  #learning rate for gradient descent
x=T.matrix(name="x")  #feature matrix
y=T.vector(name="y")  #response vector
w=theano.shared(np.zeros((d,1)),name="w")  #model parameters
risk=T.sum((T.dot(x,w).T-y)**2)/2/n  #empiricalrisk
grad_risk=T.grad(risk,wrt=w)   #gradientoftherisk
train_model=theano.function(inputs=[],outputs=risk,updates=[(w,w-learn_rate*grad_risk)],givens={x:X,y:Y})
n_steps=50
for i in range(n_steps):
    print(train_model())
print(w.get_value())

ifile.close()