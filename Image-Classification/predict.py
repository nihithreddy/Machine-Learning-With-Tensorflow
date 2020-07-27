import argparse
import os
from help import *

#Creating an object of ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument('image_path',help = 'Path of the image to be predicted')
parser.add_argument('saved_model',help = 'Path to the saved model')
parser.add_argument('--top_k',help = 'Number of Predictions to display')
parser.add_argument('--category_names',help= 'Path of the json file for mapping labels to classes')


#Parse the Arguments
args = parser.parse_args()
image_path = args.image_path
model_path = args.saved_model
top_k = args.top_k
category_names = args.category_names

#print(image_path)
#print(model_path)
#print(top_k)
#print(category_names)

#Checking all the arguments provided by the user

if(model_path!= None):
    file_name,file_extension = os.path.splitext(model_path)
    if(file_extension != ".h5"):
        print("Please provide a file with extension h5")
        exit()
        
if(top_k is None):
    top_k = 5 #By default we set the value to 5 if the value is not provided by the user
else:
    top_k = int(top_k)
    
if(category_names is None):
    category_names = "label_map.json"
    with open(category_names,"r") as f:
        class_names = json.load(f)
else:
    file_name,file_extension = os.path.splitext(category_names)
    if(file_extension != "json"):
        print("Please use a file with a json extension")
        exit()
    else:
        with open(category_names,"r") as f:
            class_names = json.load(f)
            
print(image_path)
print(model_path)
print(top_k)
print(category_names)
            
model = load_model(model_path)
probabilities,labels = predict(image_path,model,top_k)
classes = [class_names[str(label+1)] for label in labels]
print("----------- Prediction results are given below --------------")
print("The top {} predictions are {}".format(top_k,classes))
print("The Probabilities of classes are  {}".format(probabilities))

    