#Importing all the required libraries
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json

IMG_SIZE = 224
#A function that loads the model
def load_model(model_path):
    model =  tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    return model

def process_image(image):
    processed_image = tf.convert_to_tensor(image,dtype=tf.float32) #Converting numpy array to tensorflow tensor
    final_image = tf.image.resize(processed_image,(IMG_SIZE,IMG_SIZE))  #Resize the image to (224,224,3) dimension
    final_image /= 255 #Convert all the pixel values to the range(0-1)
    final_image = final_image.numpy()  #Convert the image back to a numpy array
    return final_image

#A function that takes in a image and returns the probabilities along with their class names
def predict(image_path,model,top_k):
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print("There is no file named {} in the given directory".format(image_path))
        exit()
    numpy_image = np.asarray(image)
    processed_image = process_image(numpy_image)
    processed_image = np.expand_dims(processed_image,axis=0)
    model_prediction = model.predict(processed_image)
    #Convert the model_prediction to a python list as it as a numpy array by using tolist function
    model_prediction = model_prediction[0].tolist()
    #Now we don't need all the predictions  we only need top_k predictions which is passed as an argument to the function
    #For this we will use tf.math.top_k function this function returns values and indices
    values,indices = tf.math.top_k(model_prediction,k=top_k)
    #Values are the probabilities and indices are the labels 
    #Convert them back to a python lists
    probabilities = values.numpy().tolist()
    labels = indices.numpy().tolist()
    return probabilities,labels