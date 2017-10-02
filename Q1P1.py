import numpy as np

ifile = open("linear.csv","rb")
data=np.genfromtxt(ifile,delimiter=",")
hX=data[:,1:] #all X
hY=data[:,0]  #all Y
vX=data[0:10,1:]
vY=data[0:10,0]
tX=data[10:50,1:]
tY=data[10:50,0]

print("vX shape:")
print(vX.shape)
print("vY shape:")
print(vY.shape)
print("tX shape:")
print(tX.shape)
print("tY shape:")
print(tY.shape)