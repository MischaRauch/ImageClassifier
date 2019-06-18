import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
import json

#add your arguments 
def get_inputs_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, help = "insert a int for epochs", default=3)
    parser.add_argument('-learningrate', type=int, help = "insert a int for learning rate", default=0.001)
    parser.add_argument('-numberhidden', type=int, help = "insert a int for the number of hiddenlayers", default=320)
    parser.add_argument('-model', type=str, help = "choose a training architecture, vgg or alexnet", default ="vgg16")
    parser.add_argument('-gpu', type= bool, help = "enable gpu mode with true", default=False)
    parser.add_argument('-img', type= str, help = "image path that will be classified", default ='flowers/test/10/image_07090.jpg')
    parser.add_argument('-json', type=str, help = "filepath that maps the class values to category names", default ='cat_to_name.json')
    return parser.parse_args()

#read teh file that maps the calss values to category names
def readfile(filepath):
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
#load and transform the data
def dataloader(): 
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # transforms for the training, validation, and testing sets
    train_dir_transforms = transforms.Compose([transforms.RandomResizedCrop(224), # before all 225 ##256
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
    valid_dir_transforms = transforms.Compose([transforms.Resize(224),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
    test_dir_transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, train_dir_transforms)
    valid_set = datasets.ImageFolder(valid_dir, valid_dir_transforms)
    test_set = datasets.ImageFolder(test_dir, test_dir_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size =64, shuffle=True)
    valid_data = torch.utils.data.DataLoader(valid_set, batch_size=64)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=64)
    
    return trainloader, valid_data, test_data

#download the pretrained model
def model_loading(model):
    if model == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("Error in Loading the Model: \nChoose 'vgg16' or 'alexnet' as model")
    return model

def classifier(model, hidden_units):
    #freeze parameters
    for param in model.parameters():
        param.requires_grad = False 

    #new classifier for the feedforward
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, hidden_units)), #4608,320
                                ('relu', nn.ReLU()),
                                ('dropout', nn.Dropout(0.5)),
                                ('fc2', nn.Linear(hidden_units, 300)), #320,300
                                ('relu', nn.ReLU()),
                                ('dropout', nn.Dropout(0.5)),
                                ('fc3', nn.Linear(300, 102)), #300,102
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    model.classifier = classifier
    return model

#validation function
def validation(model, valid_data, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in valid_data:
        
        images,labels = images.to('cuda'), labels.to('cuda')
        
        output = model.forward(images)
        batch_loss = criterion(output, labels)
        test_loss += batch_loss.item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

#training the model
def train(model, learningrate, epochs, gpu, trainloader, valid_data):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr =learningrate)
    
    epochs = epochs
    print_every = 40
    steps = 0
    running_loss = 0 

    if gpu == True: 
        model.to('cuda')
    for e in range(epochs):
        model.train()
        #for ii, (inputs,labels) in enumerate(trainloader):
        for inputs, labels in trainloader:
            steps += 1
        
            if gpu == True:
                inputs,labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
        
            #forward and backward 
            outputs = model.forward(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                # network in eval mode for inference
                model.eval()
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, valid_data, criterion)
                
                print("Epoch: {}/{}...".format(e+1,epochs),
                    "Loss: {:.4f}".format(running_loss/print_every),
                    "Train Loss: {:.3f}.. ".format(test_loss/len(valid_data)),
                    "Train Accuracy: {:.3f}".format(accuracy/len(valid_data)))
                running_loss = 0                    
                  
                        
            # Make sure training is back on
            model.train()
    return model

#testing
def test(model, test_data, gpu):
    correct = 0
    total = 0
    with torch.no_grad():
        for images,labels in test_data:
            #if gpu == True:
            model.to('cpu')
            #images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test_data set: %d %%' % (100 * correct / total))