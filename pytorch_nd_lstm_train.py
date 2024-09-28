#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:37:00 2024

@author: vino
A simple nD lstm regression tutorial - learning a time-series data
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import pandas as pd

###########-----Datalog-------###########
tmp=pd.read_csv(r'BTC.csv',header=[0])
tmp=tmp.iloc[-2881:]


# Custom Dataset for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        x = self.data[index:index+self.sequence_length,:]
        y = self.data[index+self.sequence_length,3:4]
        return x, y

# LSTM-MLP Model for Time Series Prediction
class LSTM_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, mlp_hidden_size):
        super(LSTM_MLP, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        mlp_out = self.mlp(lstm_out)
        return mlp_out

# Prepare the data
sequence_length = 50

#split train-test
data_tr=tmp.iloc[:2000,1:6].to_numpy()
data_ts=tmp.iloc[2000:,1:6].to_numpy()

data_tr=data_tr/np.asarray([100000,100000,100000,100000,5000000])
data_ts=data_ts/np.asarray([100000,100000,100000,100000,5000000])

dataset_tr = TimeSeriesDataset(data_tr, sequence_length)
dataset_ts = TimeSeriesDataset(data_ts, sequence_length)
dataloader_tr = DataLoader(dataset_tr,batch_size=32, shuffle=True)

# Model Parameters
input_size = 5
hidden_size = 100
num_layers = 2
output_size = 1
mlp_hidden_size = 100
learning_rate = 0.0001
num_epochs = 1000

# Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_MLP(input_size, hidden_size, num_layers, output_size, mlp_hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Training loop
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in dataloader_tr:
        #input tensor shape: [batchsize x Seq_length x input Dim], [32 x 50 x 1]
        #output tensor shape: [batchsize x Seq_length x input Dim], [32 x 1]
    
        x_batch = x_batch.view(-1, sequence_length, input_size).to(device).float()
        y_batch = y_batch.view(-1, output_size).to(device).float()

        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')

#Testing on a new sequence
model.eval()
dataloader_ts = DataLoader(dataset_ts,batch_size=1000,shuffle=False)
#test_seq = torch.tensor(data[-sequence_length:]).view(1, sequence_length, 1).to(device).float()

for x_batch, y_batch in dataloader_ts:
    x_batch = x_batch.view(-1, sequence_length, input_size).to(device).float()
    y_batch = y_batch.view(-1, output_size).to(device).float()
    
with torch.no_grad():
    prediction_ts = model(x_batch)


# Visualize
plt.figure(figsize=(10,8))
#plt.plot(data_tr, label='True tr')
plt.plot(range(len(data_ts)-sequence_length), y_batch*100000,'o', color='g', label='True ts')
plt.plot(range(len(data_ts)-sequence_length), prediction_ts*100000,'o', color='r', label='Predicted ts')
plt.legend()
plt.grid()
plt.xlim([0,50])
plt.savefig('tmp.png',dpi=300)
plt.show()



# Visualize
plt.figure(figsize=(10,8))
#plt.plot(data_tr, label='True tr')
#plt.plot(range(len(data_ts)-sequence_length), y_batch*100000,'-', color='g', label='True ts')
plt.plot(range(len(data_ts)-sequence_length), prediction_ts*100000-y_batch*100000,'-', color='r', label='Predicted ts')
plt.legend()
plt.grid()
plt.savefig('tmp2.png',dpi=300)
plt.show()


