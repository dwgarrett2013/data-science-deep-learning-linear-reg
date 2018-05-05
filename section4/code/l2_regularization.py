# we are implementing l2 regularization as code

import numpy as np
import matplotlib.pyplot as plt

# set the number of data points to 50
N = 50

# create 50 evenly spaced points between 0 and 10
X=np.linspace(0,10,N)

# y is 0.5 times x plus some random noise
Y=0.5*X + np.random.randn(N)

# manually create some outliners (set last point and 2nd last equal to 30 greater than it is)
Y[-1] += 30
Y[-2] += 30

plt.scatter(X,Y)
plt.show()

# now solving for best weights

#step 1: add the biased term
# np.ones creates an array of X elements with ones and T takes the transpose of this array
X = np.vstack([np.ones(N), X]).T

#step 2: calculate the maximum likly solution
#this is to solve for wl without the other value
w_ml=np.linalg.solve(X.T.dot(X), X.T.dot(Y))

#predictions for the solution
Yhat_ml=X.dot(w_ml)

#note, plot shows up as a line
plt.scatter(X[:,1],Y)
plt.plot(X[:,1],Yhat_ml)
plt.show()


# now perform the l2 regression
l2=1000.0

#npeye is the identity matrix
w_map=np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))

#predictions for the solution
Yhat_map=X.dot(w_map)

plt.scatter(X[:,1],Y)
plt.plot(X[:,1],Yhat_ml, label='maximum likelihood')
plt.plot(X[:,1],Yhat_map, label='map')
plt.legend()
plt.show()
