
#Author: Suryansh Kumar

#Linear regression using pytorch

#Step0: import the libraries
import torch
from torch.autograd import Variable

#Step1: data in pytorch Variable type
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

#Step2: Define the linear model class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) #1 input and 1 output
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

#Step3
model = Model()

#Step4: select the optimizer and  loss type (inbuilt libraries)
criterion  = torch.nn.MSELoss(size_average=False)
optimizer  = torch.optim.SGD(model.parameters(), lr=0.01)

#Step5: perform training
for epoch in range(2000):
    #a. perform forward pass
    y_pred = model(x_data)

    #b. compute loss
    loss = criterion(y_pred, y_data)
    print("epoch = ", epoch, "loss = ", loss.data[0])

    #c. perform backward pass
    optimizer.zero_grad()
    loss.backward()

    #d. update
    optimizer.step()

#check the optimized value of parameter
w_value, param = model.parameters()
print(w_value, param.data)

#prediction on the test data
x_test = Variable(torch.Tensor([[5.0]]))
print("prediction on the test data = ", x_test*w_value)
