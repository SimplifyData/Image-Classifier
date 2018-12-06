import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import argparse
import os
import shutil
import time

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

# Available arguments for the command line

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str , help = 'path to the data set')
parser.add_argument('--save_model' , type = str, help = 'save trained model to a checkpoint to file')
parser.add_argument('--arch', type = str, help = 'model architecure')
parser.add_argument('--learning_rate', type = float, help = 'learning rate')
parser.add_argument('--hidden_units', type = int, help = 'number of hidden units')
parser.add_argument('--epochs', type = int, help = 'number of epochs')
parser.add_argument('--gpu', action = 'store_true', help = 'utilize gpu is available - Default False')

args, _ = parser.parse_known_args()

# Training the data

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train' : transforms.Compose([transforms.RandomRotation(30),
                                  transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  normalize]),
    
    'valid' : transforms.Compose([transforms.Scale(256),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize]),

    'test' : transforms.Compose([transforms.Scale(256),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
}

if args.data_dir:
    image_datasets = {
        img_type : datasets.ImageFolder(root = args.data_dir + '/' + img_type ,transform = data_transforms[img_type])
        
        for img_type in list(data_transforms.keys())
    }

    dataloader = {
        loader_type: torch.utils.data.DataLoader(image_datasets[loader_type], batch_size=256,
                                                 shuffle=True, pin_memory=True)
        for loader_type in list(image_datasets.keys())

}

def validation(model, imgloader, criterion, device):
    test_loss = 0
    accuracy = 0
    # model.to('cuda')

    for images, labels in imgloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)

        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)

        equality = (labels.data == ps.max(dim=1)[1])

        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def train_model(image_datasets, arch='vgg16', hidden_units= 512, epochs=20, learning_rate=0.001, gpu=False, save_model =''):
    
    if args.arch:
        arch = args.arch     
        
    if args.hidden_units:
        hidden_units = args.hidden_units

    if args.epochs:
        epochs = args.epochs
            
    if args.learning_rate:
        learning_rate = args.learning_rate

    if args.gpu:
        gpu = args.gpu

    if args.save_model:
        save_model = args.save_model
        
    if arch == 'vgg16':
        # Load a pre-trained model
        model = models.vgg16(pretrained=True)

    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)

    else:
        raise ValueError('Choose model architecture vgg16 or alexnet ', arch)

    for param in model.parameters():
        param.requires_grad = False
        from collections import OrderedDict
        features = int(model.classifier[0].in_features)

        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(features, 500)),
                                                ('relu', nn.ReLU()), ('fc2', nn.Linear(500, 102)),
                                                ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    model

    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    model.to(device)

    for ii, (inputs, labels) in enumerate(dataloader['train']):

        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii ==3:
            break

    print(f'Device = {device};Time per batch: {(time.time() - start) / 3:.3f} seconds')

    #validation of test set

    epochs = epochs

    running_loss = 0

    steps = 0

    print_every = 10

    # change to cuda
    model.to(device)

    test_loss = 0

    accuracy = 0

    for e in range(epochs):
        model.train()

        for inputs, labels in dataloader['train']:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Making sure the network is in evaluation mode for interference
                model.eval()

                # Turn off gradients for validation, saves memory and computation
                with torch.no_grad():
                    valid_loss, valid_accuracy = validation(model, dataloader['valid'], criterion,device)
                    test_loss, test_accuracy = validation(model, dataloader['test'], criterion,device)

                print("Epoch: {}/{}..".format(e + 1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                "Validation Loss: {:.3f}..".format(valid_loss / len(dataloader['valid'])),
                "Validation Accuracy: {:.3f}".format(valid_accuracy / len(dataloader['valid'])),
                "Test Loss: {:.3f}..".format(test_loss / len(dataloader['test'])),
                "Test Accuracy: {:.3f}".format(test_accuracy / len(dataloader['test'])))

                running_loss = 0

                # making sure traning is back o(n

                model.train()

                print ("Our model: \n\n", model, '\n')
                print("The state dict keys: \n\n", model.state_dict().keys())

                model.class_to_idx = image_datasets['train'].class_to_idx

                print("Class to IDX: \n\n", model.class_to_idx)

                # TODO: Save the checkpoint
                checkpoint = {'input_size': 256,
                              'output_size': 10,
                              'class_to_idx': model.class_to_idx,
                              'state_dict': model.state_dict()}
                if save_model:
                    print ('storing model to checkpoint: ', checkpoint)
                    torch.save(checkpoint, save_model)

                return model

train_model(image_datasets)