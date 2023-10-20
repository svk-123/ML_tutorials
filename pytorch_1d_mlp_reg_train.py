#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 19:52:00 2023

@author: vino
A simple 1D mlp regression tutorial - learning a wiggly fucntion
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

############################################
### consider a wiggly function 
###(Nielsen M. A., (2015), Neural Network and Deep Learning. Determination Press)
###f(x) = 0.2 + 0.4x 2 + 0.3xsin(15x) + 0.05cos (50x)
############################################

#train samples
x=np.linspace(0,1.0,1000)
y=0.2+0.4*x**2+0.3*x*np.sin(15*x)+0.05*np.cos(50*x)

### train-validate split
np.random.seed(123)
I=np.arange(len(x))
np.random.shuffle(I)

N=800
x_train=x[I[:N]]
y_train=y[I[:N]]

x_val=x[I[N:]]
y_val=y[I[N:]]

#test samples
x_test=np.linspace(1,1.2,100)
y_test=0.2+0.4*x_test**2+0.3*x_test*np.sin(15*x_test)+0.05*np.cos(50*x_test)

plt.figure()
plt.plot(x_train,y_train,'o')
plt.plot(x_val,y_val,'o')
plt.plot(x_test,y_test,'o')
plt.show()

###############################################
### Data loader 
###############################################
# transform to torch tensor
#reshape np array to (N,dim)
#if 1D, then (N,1)
x_train = torch.Tensor(x_train.reshape(-1,1)) 
y_train = torch.Tensor(y_train.reshape(-1,1))

x_val = torch.Tensor(x_val.reshape(-1,1)) 
y_val = torch.Tensor(y_val.reshape(-1,1))

x_test = torch.Tensor(x_test.reshape(-1,1)) 
y_test = torch.Tensor(y_test.reshape(-1,1))

# create datset
train_dataset = DataLoader(TensorDataset(x_train,y_train),batch_size=64, shuffle=True)
val_dataset = DataLoader(TensorDataset(x_val,y_val)) 

#######check batch size & shape #########
x_tmp, y_tmp = next(iter(train_dataset))
print('batch size',x_tmp.shape)
print('num. batches', len(train_dataset))

###############################################
######## check hardware###########################
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

###############################################
######## Build model ###########################
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        output = self.mlp(x)
        return output
    
model = NeuralNetwork().to(device)
print(model)

######### Hyper parameters#####################
learning_rate = 1e-3
batch_size = 64
epochs = 5


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if batch % 100 == 0:
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"training   loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def val_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    val_loss= 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            val_loss += loss_fn(pred, y).item()

    #val_loss /= num_batches

    print(f"validation loss: {val_loss:>8f} \n")


# Initialize the loss function
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataset, model, loss_fn, optimizer)
    val_loop(train_dataset, model, loss_fn)
print("Done!")


#################################################################
###########save model###########################################
torch.save(model,'./model/1d_model')