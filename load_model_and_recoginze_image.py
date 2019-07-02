
import torch 
from torchvision import  models
import torch.nn as nn
from torch import optim

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from PIL import Image
import json
from helper import process_image


def load_checkpoint(checkpoint):
    '''bulid the model from it's structre and checkpoint saved state_dict
    input,pre-saved model checkpoint 
    reuturn, trained model
    '''
    model = models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(nn.Linear(2048, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(512, 102),
                                     nn.LogSoftmax(dim=1))
    model.fc= classifier
    print('loading model architecture finished')
    
    checkpoint = torch.load(checkpoint,map_location = 'cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    print('load state-dict from checkpoint  finished')
    return model



def predict_topk(image_path, model,mapping_json, gpu,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #image showing
    print('read in cata-to-name files')
    with open(mapping_json, 'r') as f:
        cat_to_name = json.load(f)
    # gpu or 
    print('reading in file finished')
    if gpu:
        model.cuda()
    else:        
        model.cpu()
    model.eval()
    #image prediction
    image = process_image(image_path)
    model = model.double()
    image = image.transpose((2,1,0))
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    with torch.no_grad():        
        log_ps = model(image)
        ps = torch.exp(log_ps)                
        probs,top_class = ps.topk(5,dim = 1)
        probs,top_class = probs.numpy(),top_class.numpy()
    #show the top5 flowers name    
    flower_names= [cat_to_name[class_] for class_ in classes[0].astype(str)]
    
    for prob,flower_name in zip(probs,flower_name):
        print('the probablity of ', flower_name,prob,' is ', prob*100, '%')      
    return probs,top_class

    
    



