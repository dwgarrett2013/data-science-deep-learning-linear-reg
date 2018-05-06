# shows workaround for for dummy variable trap

import numpy as np
import matplotlib.pyplot as plt

#set number of data points to 10
N=10
#set dimensionality to 3
D=3

#initialize X as and NxD matrix
X=np.zeros((N,D))

# set the bias terms
X[:,0]=1

#set first 5 elements of first column are equal to 1
X[:5,1]=1

#set last 5 elements of 2nd column to 1
X[5:,2]=1

#show that the entire 0th column and the remaining results
print X

#set first 5 outpouts to be 0 and second 5 to be 1
Y=np.array([0]*5 + [1]*5)

#show that it worked
print Y

#see that if we try to do the regular linear algebra solution, it will fail
#w=np.linalg.solve(X.T.dot(X), X.T.dot(Y))

#we have to use gradient descent instead

#create an array that shows the dropping costs
costs=[]

#This assigns random weights
#also ensures that has variace of 1/D (to align with gausian distribution)
w=np.random.randn(D) / np.sqrt(D)

learning_rate=0.001

for t in xrange(1000):
    Yhat=X.dot(w) #this calculates our predictions
    delta=Yhat-Y #this is the difference betwen Yhat and Y
    w=w-learning_rate*X.T.dot(delta) #this is explained on pg. 24 of handwritten notes
    mse=delta.dot(delta) /N #this calculates the mean squared error using cross product and finds the mean of the resulting set
    costs.append(mse)

#plot the costs and shows that the cost drops with every iteration of gradient descent
plt.plot(costs)
plt.show()

#print the final w or the solution to this problem.  This got better over our many different iterations
print w

#we can confirm the solution by plotting Yhat and Y and it shows that the predictions are very close to the targets
plt.plot(Yhat, label='prediction')
plt.plot(Y, label='targets')
plt.legend()
plt.show()
