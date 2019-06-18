from calcutrain import get_inputs_args
from calcutrain import dataloader
from calcutrain import model_loading
from calcutrain import classifier
from calcutrain import train
from calcutrain import test
import json

#set parameters with command line
in_arg = get_inputs_args()
print(in_arg)


#load the data
trainloader, valid_data, test_data = dataloader()

#label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#laod the pretrained model
model = model_loading(in_arg.model)
print("Done loading the model")

#create classifier but befor check hidden units
full_model = classifier(model,in_arg.numberhidden)
print("Done building the classifier")

#train the NN
training = train(full_model, in_arg.learningrate, in_arg.epochs, in_arg.gpu, trainloader, valid_data)
print("Done with Training")
#test on test_data
test = test(training, test_data, in_arg.gpu)