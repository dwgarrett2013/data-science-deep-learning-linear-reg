# This file will generate a line of best fit for an X and Y coordinate plane with 2 variables

#import numpy and matplotlib
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
# load the data from csvs

# create blank arrays to store the values from the .csv file
X = []
Y = []

for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))  # convert X to a float and append it to array X
    Y.append(float(y))  # convert Y to a float and append it to array Y


# turn arrays X and Y into numpy arrays
X=np.array(X)
Y=np.array(Y)

# plto the data to see what it looks like
#plt.scatter(X,Y)
#plt.show()

# apply the equations we learned to calculate a and b
# Sumof(Xi*Xi)=Xsquared=X.dot(x)
# xbar=X.mean(x)
denominator = X.dot(X) - X.mean()*X.sum()

#numerator
# dot product is the sum of everything in one times another, so summation(XiYi) = X.dot(Y)
# https://en.wikipedia.org/wiki/Dot_product
a= ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b= ( (Y.mean()*X.dot(X)) - (X.mean()*X.dot(Y)) ) / denominator

# calculated predicted Y
Yhat = (a*X) + b

# Plot the result on the same scatter plot
plt.scatter(X,Y)
plt.plot(X,Yhat)
#plt.show()


# calculating R-Squared
d1 = Y - Yhat   # creates a vector of the differences between expected and results, this is used to find SSres (still need to square and sum)
d2 = Y - Y.mean()  # this is a vector used to find  SStotal (still need to square and sum)
r2 = 1 - (d1.dot(d1)/d2.dot(d2))    #input the rsqaured formula

print "the r-squared is:", r2
plt.show()
