# this is an example of gradient descent
import numpy as np

#set the initial value of w
w=20

# loop through the functin 30 times
for i in xrange(30):
    w=w-0.1*2*w
    #print w

# We can see that 30 is not enough so, lets try a bigger number

#reset the initial value of w
w=20

# loop through the functin 100 times
for i in xrange(100):
    w=w-0.1*2*w
    #print w

# by moving slowly in the direction of the gradient (or derivative of a function
# we can get closer and closer to the minimum of a function


#Extra example
# note you need to take the derivative of the initial function, set to zero, and then find optimal values

w1=.75
w2=.75
learning_rate=.3

for i in xrange(1000):
    w1-=learning_rate*(2*w1)
    w2-=learning_rate*(4*(w2**3))
    print w1,w2
