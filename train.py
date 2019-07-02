

# Train on miages data with transfer learning architecture
''' print out the traing loss,validation loss ,and loss accuracies while training

Options:
- Choose pretrained model architecture, python train.py --arch 'resnet101'
- Adust on hyperparameters, python train.py data_dir --learning_rate 0.0001 --hidden_units 512 --eopches 12
- Shift on GPU for traing ,python train.py data_dir --cuda
- Choose directroy to savming the check point

Example usage:
 python train.py flowers --gpu --save_dir
'''
# libraries

import argparse
import numpy as np

from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
from torch import optim

import time
import copy
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from workspace_utils import active_session


from preprocess_data import preprocess_data
from train_model import load_pre_trained_model, nn_classifer_train_valid,save_check_point

# Get the command line inputs
parser = argparse.ArgumentParser()

# Basic usage: python train.py data_directory
parser.add_argument('--data_directory', action='store',
                    default = 'flowers',
                    help='choose data directory to load training data, e.g., "flowers"')

# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
parser.add_argument('--save_dir', action='store',
                    default = 'resnet101_eopch_4.pth',
                    dest='save_dir',
                    help='Select directory to saving checkpoints, e.g., "./assets/filename.pth"')

# Choose pretrained architecture: python train.py data_dir --arch "resnet101"
parser.add_argument('--arch', action='store',
                    default = 'resnet101',
                    dest='arch',
                    help='Choose architecture, e.g., "resnet101"')

# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
parser.add_argument('--learning_rate', action='store',
                    default = 0.001,
                    dest='learning_rate',
                    help='Choose optimizer learning rate, e.g., 0.0001')
# set the hiddent units
parser.add_argument('--hidden_units', action='store',
                    default = 512,
                    dest='hidden_units',
                    help='Choose architecture hidden units, e.g., 512')
# training epoches
parser.add_argument('--epochs', action='store',
                    default = 1,
                    dest='epochs',
                    help='Choose architecture number of epochs, e.g., 20')

# Use GPU for training: python train.py data_dir --gpu
parser.add_argument('--gpu', action='store_true',
                    default=True,
                    dest='gpu',
                    help='Use GPU for training, set a switch to true')

parse_results = parser.parse_args()


# parse  
data_dir = parse_results.data_directory
save_dir = parse_results.save_dir
arch = parse_results.arch
learning_rate = float(parse_results.learning_rate)
hidden_units = int(parse_results.hidden_units)
epochs = int(parse_results.epochs)
device = parse_results.gpu


#load data and preprocessing as well
image_datasets,train_loader,valid_loader,test_loader = preprocess_data(data_dir)
#bulid the pre_trained model structure
model_init,optimizer= load_pre_trained_model(arch, hidden_units)
#train the model
model,validation_accuracies  = nn_classifer_train_valid(epochs,model_init,optimizer,train_loader,valid_loader,device)
#saving the checkpoint
check_point = save_check_point(model,image_datasets['train'],save_dir)
 



