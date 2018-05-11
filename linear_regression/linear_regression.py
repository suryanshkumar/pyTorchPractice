#Author: Suryansh Kumar

#Linear regression cost on a linear data

#import the essential library
import numpy as np
import matplotlib.pyplot as plt

#Step 1. define forward prediction model (In this example its linear)

#Step 2. define the loss function

#Step1: Definition: Given the input x and parameter w, predict y_pred as y_pred = x*w
def predict(x, w):
    return x*w

#Step2: define the loss with respect to the recent prediction (y_pred-y_true)^2/2
def loss(x, y_true, w):
    y_pred = predict(x, w)
    loss = 0.5*(y_true-y_pred)*(y_true-y_pred);
    return loss

#Step3: Assuming the data's are provided, start with some random guess of the parameter and check the mean square error
def estimate_mse(x_data, y_data, w, N):
    total_error = 0
    for x_val, y_val in zip(x_data, y_data):
        loss_val = loss(x_val, y_val, w)
        total_error = total_error + loss_val
        #print("x_data, y_data, loss", x_val, y_val, loss_val)
    mse = total_error/N
    return mse

#Step4: Check the error using some basic data-set with range of w's and plot the Error curve
x_data = [1, 2, 3, 4]
y_data = [2, 4, 6, 8]
N = len(y_data)
mse_error= []
wei_val  = []

for w in np.arange(0.0, 4.5, 0.5):
    mse = estimate_mse(x_data, y_data, w, N)
    print("Mean Square Error with parameter w = " +str(w) + " is " + str(mse))
    mse_error.append(mse)
    wei_val.append(w)
plt.plot(wei_val, mse_error, 'ro-')
plt.xlabel("parameter (w) value")
plt.ylabel("Mean Squared Error Loss")
plt.title("Error prediction curve")
plt.show()

#Summary: In this example I changed the value of w and can see that the error value is minimum for w = 2
#However, it would be better if a program can automatically figure out the value for which the error is minimum
#which motivated me to investigate as to how I can find this 'w' value for which the error will be minimum
#Possibly, I have to go through the theory of mathemantical optimization to figure this out.
#In the next assignment I will code ONE of the basic way to estimate optimal 'w' for this task.
