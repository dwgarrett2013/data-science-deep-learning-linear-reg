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

#step 2: calculate the maximum liklihood solution (using standard form of polynomial linear regression)
#This is of form w=(X^TX)^(-1)  *  X^Y
# to note regarding X.T.dot(https://stackoverflow.com/questions/42517281/difference-between-numpy-dot-and-a-dotb)
w_ml=np.linalg.solve(X.T.dot(X), X.T.dot(Y))

#step 3: Predict the solution using the weights calculated in step 2
Yhat_ml=X.dot(w_ml)

#note, plot the original scatter plot and the maximium likelihood (normal polynomial regression) line
plt.scatter(X[:,1],Y)
plt.plot(X[:,1],Yhat_ml)
plt.show()


#now perform the l2 regression
# probably don't need an L2 regularization this high in many problems
# everything in this example is exaggerated for visualization purposes

#this is an arbitrary penalty that the user chooses to set to penalize high weights
# L2 = lambda
l2=1000.0

# Step 5 calculate the weights using the data
#np.eye(2) forms a 2x2 identity matrix with 1s on the main diagnoal
#l2*np.eye(2) forms a 2x2 matrix with lambdas (L2) across the diagonal means that the penalty will be applied to each weight
#https://www.geeksforgeeks.org/numpy-eye-python/ explains np.eye()
w_map=np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))

# Step 6 use the weights from step 5 to make a prediction
Yhat_map=X.dot(w_map)

#plot the final results which include a scatter plot of all the points, the maximum likelihood (unpenalized polynomial regression), the l2 polynomial regression, and the legend

plt.scatter(X[:,1],Y)
plt.plot(X[:,1],Yhat_ml, label='maximum likelihood')
plt.plot(X[:,1],Yhat_map, label='map')
plt.legend()
plt.show()
