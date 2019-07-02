

# using trained checkpoin to identfiy image
''' print out the top 5 possibilities species of one image

Options:
- Load the trained modle form directory, python predict.py --save_dir './assests'
- Choose image data from directory, python train.py --image './flowers/test/1/'
- Show top 5 possibility for image reconginization --topk 5


Example usage:
 python predict  --save_dir --image
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
from workspace_utils import active_session
import matplotlib.pyplot as plt


from load_model_and_recoginze_image import load_checkpoint,predict_topk



# Get the command line inputs
parser = argparse.ArgumentParser()


# Basic usage: python train.py data_directory
parser.add_argument('--image_path', action='store',
                    default = 'flowers/test/10/image_07090.jpg',
                    help='choose data directory to load training data, e.g., "./flowers/image_name"')

# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
parser.add_argument('--checkpoint_dir', action='store',
                    default = 'resnet101_eopch_.pth',
                    dest='checkpoint_dir',
                    help='Select  saving checkpoints to load the model, e.g., "./resnet101_eopch_.pth"')

# Use GPU for training: python train.py data_dir --gpu
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for training, set a switch to true')
# Choose pretrained architecture: python train.py data_dir --arch "resnet101"
parser.add_argument('--topk', action='store',
                    default = 5,
                    dest='topk',
                    help='Choose architecture, e.g., 5')
# clas to name jason file 
parser.add_argument('--mapping_json', action='store',
                    default = 'cat_to_name.json',
                    dest='mapping_json',
                    help='label to name json file, e.g., "cat_to_name"')


parse_results = parser.parse_args()
# parse  
image_path = parse_results.image_path
checkpoint_dir = parse_results.checkpoint_dir
topk = int(parse_results.topk)
mapping_json = parse_results.mapping_json
gpu = parse_results.gpu

#import the model from checkpoint
model = load_checkpoint(checkpoint_dir)
#predict the image class and show the top5 of those
probs,top_classes= predict_topk(image_path, model, gpu,topk,mapping_json)

 



