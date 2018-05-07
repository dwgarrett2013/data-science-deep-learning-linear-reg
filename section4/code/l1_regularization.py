#this will perform 1 regularization on a function using gradient descent
#goal: generate data as "fat" matrix and use l1 regularization to assign weights that are appropriate (0 or not 0) given what is noise

import numpy as np
import matplotlib.pyplot as plt

#create a fat matrix
N=50
D=50

#X is uniformly distributed points of size NxD centered around zero from -5 to 5
#remember X is just your point values, weights are elsewhere
X=(np.random.random((N,D)) - 0.5)*10

#set true 2 to be 1, 0.5, -0.5 then everything else is zero (last 3 terms do not influence the output)
true_w=np.array([1,0.5, -0.5] + [0]*(D-3))

#numpy.random.randn returns a sample from the standard normal distribution
#parameters are dimentions of the array)
#mulitply by randomn noise by 0.5

#in all this returns an array of targets that have a slight amount of noise
Y=X.dot(true_w) + np.random.randn(N)*0.5

#we perform gradient descent using l1 to get the result
costs=[]

#this can be a standard value for the Ws
w=np.random.randn(D) / np.sqrt(D)

#this is the learning rate of the algorithm
learning_rate=0.001

#this is an arbitrary penalty that the user chooses to set to penalize high weights
# L1 = lambda
l1=10.0

#run the gradient descent using what we had in our other gradient descent problem
for t in xrange(500):
    Yhat=X.dot(w)
    delta=Yhat -Y

    #we use the funtion from pg. 25 on notes to accentuate the values
    #
    w=w-learning_rate*(X.T.dot(delta)+ l1*np.sign(w))

    mse=delta.dot(delta) /N #this calculates the mean squared error using cross product and finds the mean of the resulting set
    costs.append(mse)

plt.plot(costs)
plt.show()

print "final w:",w

plt.plot(true_w, label='true_w')
plt.plot(w, label='w_map')
plt.legend() #so we know which is which
plt.show()
