import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

import argparse
import os
import shutil
import time

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import json
import argparse

from PIL import Image
from collections import OrderedDict

# Command line args
parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, help='Path of image to predict')
parser.add_argument('--checkpoint', type =str , help = 'Path of the save model check point')
parser.add_argument('--top_k', type = str , help = 'top K most likely classes')
parser.add_argument('--category_names', type = str, help = 'Mapping of categories to real names')
parser.add_argument('--gpu', action = 'store_true' , help = 'IF GPU is turned on or off - True or False')

args, _ = parser.parse_known_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #size = 256, 256
    #box = 16,16, 240, 240
    
    
    pil_im = Image.open(image)
    #im.thumbnail(size)
    #im.crop(cr_size)
    #np_image = np.array(im)
    
    transform_images = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    
    pil_im = transform_images(pil_im).float()
    
    mean= np.array([0.485, 0.456, 0.406])
    
    std = np.array([0.229, 0.224, 0.225])
    
    processed_image = np.array(pil_im)
    
    #normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    
    processed_image = (np.transpose(processed_image, (1,2,0)) - mean) / std
    
    #processed_image = np.transpose(processed_image, (1,2,0))
    
    
    #processed_image = normalize(processed_image)
    
    #processed_image.Normalize(mean = mean , std = std)
    
    processed_image = np.transpose(processed_image, (2,0,1))
    
    return processed_image

# Predicting class of the flower from the image

def predict(image_path, checkpoint, top_k = 5, category_names = '' , gpu = False):
    ''' Predict the most likley class of an image provided its labels using deep neural network
    '''
          
    if args.top_k:
        top_k = args.top_k
    
    if args.category_names:
        category_names = args.category_names
    
    if args.gpu:
        gpu = args.gpu

    #loading saved model
    checkpoint = torch.load(checkpoint)
    model = models.vgg16(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
        features = int(model.classifier[0].in_features)
    
        classifier = nn.Sequential(OrderedDict([ ('fc1', nn.Linear(features, 500)),
                                                ('relu', nn.ReLU()), ('fc2', nn.Linear(500, 102)), 
                                                ('output', nn.LogSoftmax(dim=1)) ]))

    model.classifier = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    model
    
    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'
        
    model.to(device)
    
    model.eval
    
    image = Variable(torch.FloatTensor(image_path), requires_grad = True)
    
    # For VGG
    image = image.unsqueeze(0)
    
    image = image.to(device)
    
    pred = model.forward(image).topk(int(top_k))
    
    #cacculate the class probablities (softmax for img)
    if gpu:
        with torch.no_grad():
            prob = F.softmax(pred[0].data , dim = 1).cpu().numpy()[0]
            classes = pred[1].data.cpu().numpy()[0]
    else:
        probs = torch.nn.functional.softmax(pred[0].data, dim=1).numpy()[0]
        classes = pred[1].data.numpy()[0]
        
        
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
            
            
        labels = list(cat_to_name.values())
        classes = [labels[x] for x in classes]
    
    model.train
    
    return prob,classes

if args.img_path and args.checkpoint:
    image_path = process_image(args.img_path)
    prob, classes = predict(image_path, args.checkpoint)
    print(prob)
    print(classes)

    
    
