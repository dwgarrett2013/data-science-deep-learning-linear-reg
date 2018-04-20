#This file will demonstrate moore's law in python code

import re #this is regex to remove nondecimal characters
import numpy as np
import matplotlib.pyplot as plt

X=[]
Y=[]

non_decimal=re.compile(r'[^\d]*')   #this is a regex pattern to get rid of bad data
#non_decimal=re.compile(r'[^\d]+')

for line in open('moore.csv'):
    r=line.split('\t') # split the line on tabs

    x=int(non_decimal.sub('', r[2].split('[')[0])) #remove everything that is not a decimal.  Want everything to the left of the square bracket in column 3
    y=int(non_decimal.sub('', r[1].split('[')[0])) #remove everything that is not a decimal.  We want transistor colummn (column 2)
    #add both values to their respective vectors
    X.append(x)
    Y.append(y)

#convert to numpy arrays
X=np.array(X)
Y=np.array(Y)

#show the graph
plt.scatter(X,Y)
plt.show()

# taking log of y shows that log is a linear line (this may just be to show that moore's laws is logarithmic)
# https://stackoverflow.com/questions/28058527/can-some-one-explain-me-what-does-np-log-do
Y = np.log(Y)
plt.scatter(X,Y)
plt.show()

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

#want to determine how go the model is using r-Squared

d1=Y-Yhat
d2=Y-Y.mean()

#find rsquared using SSres and SStot
r2=1-(d1.dot(d1)/d2.dot(d2))

#print values
print("a:",a,"b:",b)
print("the r-squared is:",r2)

# To find the time to double (year1 to year 2)

# tc=transistor count
# exp(x) is e^x power

# log(transistor count) = a*year +b
# tc=exp(b)*exp(a*year)
# 2*tc = 2 *exp(b) * exp(a*year) = exp(ln(2)) * exp(b)*exp(a*year)
#   using ln, we can get 2 into the numerator
#   = exp(b) * exp(a * year + ln(2))
# exp(b)*exp(a*year2) = exp(b)*exp(a*year1 + ln2)
# a*year2 = a*year1 +ln2
# year2=year1 +ln2/a
print("time to double:",np.log(2)/a, "years")

plt.show()
