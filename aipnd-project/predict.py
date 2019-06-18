from calcutrain import get_inputs_args
from calcutrain import dataloader
from calcutrain import model_loading
from calcutrain import classifier
from calcutrain import test
from calcutrain import readfile
from calcupredict import load_checkpoint
from calcupredict import predict
from calcupredict import sanity_check

#create the classifier and assign the correct checkpoint
    #get user input
in_arg = get_inputs_args()
    #load the data
trainloader, valid_data, test_data = dataloader()
    #label mapping
cat_to_name = readfile(in_arg.json)
    #laod the pretrained model
model = model_loading(in_arg.model)
print("Done loading the model")
    #create classifier 
full_model = classifier(model,in_arg.numberhidden)
print("Done building the classifier")
    #assign the checkpoint 
load_checkpoint = load_checkpoint('checkpoint.pth', full_model, in_arg.numberhidden)
print("Done loading the checkpoint into the model")

#testing the checkpoint
#test = test(load_checkpoint, test_data, in_arg.gpu)

#predict most likely image class and it's probability
image = in_arg.img
predict = predict(image, load_checkpoint)
#get the right accuracy
solution_array = predict[0]

#sanity check
sanity_check = sanity_check(image, test_data, load_checkpoint, cat_to_name)
print("\nThe Image is a:", sanity_check[0], "with an:", solution_array[0],"% accuracy")

#print out the top K classes with probability
print("\nThe top k classes with assosiated probability are:")
print(sanity_check[0], "with an:", solution_array[0], "% accuracy")
print(sanity_check[1], "with an:", solution_array[1], "% accuracy")
print(sanity_check[2], "with an:", solution_array[2], "% accuracy")
print(sanity_check[3], "with an:", solution_array[3], "% accuracy")
print(sanity_check[4], "with an:", solution_array[4], "% accuracy")