#Author: Suryansh Kumar
#Second assignment: Find the value of parameter that minimizes the loss
#w := w - \alpha*(gradloss/gradw)

import numpy as np
import matplotlib.pyplot as plt

#Step 0: training data
x_data = [1, 2, 3, 4]
y_data = [2, 4, 6, 8]
eta = 0.01 #learning rate
#Step 0: test data
x_test = 7.0

#Step1: prediction
def predict(x, w):
    return x*w

#Step2: loss function
def loss(x, y_true, w):
    y_pred = predict(x, w)
    loss = 0.5*(y_true-y_pred)*(y_true-y_pred);
    return loss

#Step3: compute gradient w.r.t w
def gradient(x, y):
    return x*(x*w-y)


#Step4: Assuming the data's are provided, start with some random guess of the parameter and check the mean square error
def estimate_mse_update_w(x_data, y_data, w, N):
    total_error = 0
    for x_val, y_val in zip(x_data, y_data):
        w = w-eta*gradient(x_val, y_val)
        loss_val = loss(x_val, y_val, w)
        total_error = total_error + loss_val
    mse = total_error/N
    return mse, w

N = len(y_data)
mse_error= []
wei_val  = []
w = 1.0 #start with some random value


#step5: predict, estimate error and update the parameter over iteration
for i in range(200):
    mse, w = estimate_mse_update_w(x_data, y_data, w,  N)
    #print("Mean Square Error with parameter w = " +str(w) + " is " + str(mse))
    mse_error.append(mse)
    wei_val.append(w)
    if (mse<1e-10):
        w_optimal = w;
        e_at_optimal = mse
        print("optimal parameter value = " + str(w_optimal))
        break
    else:
        continue

#predict the value of y on the test sample
y_test = w_optimal*x_test
print("The predicted value for " +str(x_test) + " is " + str(y_test))

#plot the convergence curve
plt.plot(wei_val, mse_error, 'ro-', w_optimal, e_at_optimal, 'go')
plt.xlabel("parameter (w) value")
plt.ylabel("Error prediction curve")
plt.title("Convergence curve")
plt.show()
