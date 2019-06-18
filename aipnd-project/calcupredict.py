import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt

#load the checkpoint
def load_checkpoint(filepath, model, hidden_units):
    
    #turn off param.requires_grad
    for param in model.parameters():
        param.requires_grad = False
    #load classifier and the rest of nessecarry date to rebuild the NN    
    checkpoint = torch.load(filepath)
    model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(checkpoint['input_size'], hidden_units)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(0.5)),
                            ('fc2', nn.Linear(hidden_units, 300)), #320,300
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(0.5)),
                            ('fc3', nn.Linear(300, checkpoint['output_size'])),
                            ('output', nn.LogSoftmax(dim=1))
                            ])) 
   
    model.load_state_dict(checkpoint['state_dict'])
    #model.load_optimizer.state_dict(['optimizer'])
    model.learning_rate = checkpoint['learning_rate']
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']

    print("Done loading the model")    
    return model

#process the image in the right dimensions
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_img = Image.open(image)
    
    #pil_img = pil_img.resize((256,256))
    #pil_img = pil_img.crop((0,0,224,244)) return the image in wrong sizes. with support form student hub mentor:
    
    #resize
    current_width, current_height = pil_img.size
    if current_width < current_height:
        new_height = int(current_height * 256 / current_width)
        pil_img = pil_img.resize((256, new_height))
    else:
        new_width = int(current_width *256 / current_height)
        pil_img = pil_img.resize((new_width, 256))
    
    #crop image
    precrop_width, precrop_height = pil_img.size
    left = (precrop_width -224)/2
    top = (precrop_height -224)/2
    right = (precrop_width +224)/2
    bottom = (precrop_height +224)/2
    
    pil_img = pil_img.crop((left,top,right,bottom))
    
    
    np_img = np.array(pil_img)
    
    np_img = np.array(np_img)/255
    
    means = [0.485,0.456,0.406]
    std = [0.229,0.224,0.225]
    
    np_img = (np_img - means) /std
    
    np_img = np_img.transpose((2,0,1))
    
    return torch.Tensor(np_img)#torch.Tensor(np_img) #image

#calculate prediction of an image
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to('cuda')
    with torch.no_grad():
        image = process_image(image_path)
        image = image.to(device)
        #needed for something with the batch not sure exactly but code works :)
        image = image.unsqueeze_(0)
        
        #image predictions
        image_pred = model.forward(image)
        
        #return the top k classes
        p, classes = torch.topk(image_pred, topk)
        p = p.exp() # calc all exponential of all elements
        class_to_idx = model.class_to_idx        
        
        #avoiding errors with this
        p = p.cpu().numpy()
        classes = classes.cpu().numpy()
        
        #put the indexes in numerical order
        classes_indexed = {model.class_to_idx[i]: i for i in model.class_to_idx}
        #class must be a list!
        classes_list = list()
    
        for label in classes[0]:
            classes_list.append(classes_indexed[label])
        
        return (p[0], classes_list)
    
#sanity check
def sanity_check(image,test_set,model,cat_to_name):
    
    result = process_image(image)
    #ax = imshow(result, ax)
    #ax.axis('off')

    #ax.set_title(cat_to_name[str(test_set)])


        # Make a prediction on the image
    predictions, classes = predict(image, model)
    
        # Get the lables from the json file
    labels = []
    for c in classes:
        labels.append(cat_to_name[str(c)])

    return labels
