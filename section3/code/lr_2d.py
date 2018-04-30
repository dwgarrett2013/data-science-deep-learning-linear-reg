## this is my file to run 2d linear regression

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# load the data
X = []
Y=[]

# read every line in the file
# first column is x1, 2nd column is x2, third column is y
for line in open('data_2d.csv'):
    x1, x2, y=line.split(',')
    X.append([float(x1), float(x2), 1])   ## X0 is one all the time.  this is implicit in the solution because we don't have a bias when using actual data
    Y.append(float(y))

# turn X and Y in to numpy arrays
X=np.array(X)
Y=np.array(Y)

# plot the data to see what we have
# this creates a 3d scatterplot
#fig=plt.figure()
#ax=fig.add_subplot(111,projection='3d')
#ax.scatter(X[:,0], X[:,1], Y)
#plt.show()

#calculate the weights
#note: in numpy '*' does element.element, np.dot() does matrix multiplication
# X.T means X transpose T

#this finds the weights and stores in w
#this takes into account the inverse
w=np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

# this finds Yhat by multiplying X (inputs) and w(weights)
Yhat=np.dot(X,w)

# computer r-squared
d1=Y-Yhat
d2=Y-Y.mean()
# find square of Yhat residuals over square of mean residuals
r2=1-d1.dot(d1) / d2.dot(d2)
print "r-squared:",r2
