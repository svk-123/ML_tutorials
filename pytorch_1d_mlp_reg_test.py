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

#test the model for train/test samples

