#this file will show how to do linear regression on a polynomial function

import numpy as np
import matplotlib.pyplot as plt

# load the data
X=[]
Y=[]
for line in open('data_poly.csv'):
    x,y = line.split(',')
    x=float(x)
    #note a difference from video (the weight variable of 1 will come at the end)
    #from there load arrays, first term is constant, second is x, third is xsquared to follow the format of the ds
    # remember the coefficients are all that is being determined,so the way these are added will generate a coefficient for each field regardless of how it is calculated
    X.append([1, x, x*x])
    Y.append(float(y))

#convert to numpy arrays
X=np.array(X)
Y=np.array(Y)

#practice plot
# colon for all rows and 1 for first column (means second column in pytho)
plt.scatter(X[:,1], Y)
plt.show()

# calculate weights
# this takes into account the inverse
w=np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat=np.dot(X,w)

#plot it all together
plt.scatter(X[:,1],Y)
#will need the sorted to ensure points are in the order that they are suppsed to be

plt.plot(sorted(X[:,1]), sorted(Yhat))
plt.show()

# computer r-squared
d1=Y-Yhat
d2=Y-Y.mean()
# find square of Yhat residuals over square of mean residuals
r2=1-d1.dot(d1) / d2.dot(d2)
print "r-squared:",r2
