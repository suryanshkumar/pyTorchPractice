#Author: Suryansh Kumar

#Step0: import the essential library
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

#Step1: Load data using dataloader
class Diabetes_Dataset(Dataset):
    def __init__(self):
        data = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
        self.len = data.shape[0]
        self.train_data = torch.from_numpy(data[:, 0:-1])
        self.train_label = torch.from_numpy(data[:, [-1]])

    def __getitem__(self, index):
        return self.train_data[index], self.train_label[index]

    def __len__(self):
        return self.len

dataset = Diabetes_Dataset()
train_loader = DataLoader(dataset=dataset, batch_size=40, shuffle=True)

#Step2 Define model class (In this example  3 layers)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = torch.nn.Linear(8, 6)
        self.layer2 = torch.nn.Linear(6, 4)
        self.layer3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        output1 = self.sigmoid(self.layer1(x))
        output2 = self.sigmoid(self.layer2(output1))
        y_pred  = self.sigmoid(self.layer3(output2))
        return y_pred

model = Model()

#Step3: Select the inbuilt loss and optimizer
criterion =  torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

#Step4: train the network
#Follow the rythm 1. do prediction on present parameters(forward) 2. estimate loss 3.compute gradient backward, update the parameter

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        input_data, input_label = data
        input_data, input_label = Variable(input_data), Variable(input_label)

        #1 .Forward pass
        y_pred = model(input_data)

        #2. estimate loss
        loss = criterion(y_pred, input_label)
        print(epoch, i, loss.data[0])

        #3. backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
