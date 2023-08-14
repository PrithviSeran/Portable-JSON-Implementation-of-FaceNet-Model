import tensorflow as tf
from keras.models import model_from_json
import numpy as np

"""
Create a folder with a picture of each person who you want the model to identify. 
"""


#loading model from json file
json_file = open('facenet/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

#turning the file into the model 
model = model_from_json(loaded_model_json)

#loading the weights of the model
model.load_weights('facenet/model.h5')


list_of_people = [""" Enter all the names in your folder  """]

#turning the image into the encoded version (taken from Deep Learning Specialization)
def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)

    return embedding / np.linalg.norm(embedding, ord=2)


#finding who is the most probable person in the picture
def who_is_it(image_path):

    lowest_dist = 100

    person = img_to_encoding(image_path, model)

    #comparing the distances with the inputed image and the images in the folder 
    for name in list_of_people:
        data_encoding = img_to_encoding("""f "the path of the folder with the pictures goes here "/{name}.jpeg""", model)

        dist = np.linalg.norm(person - data_encoding)

        #saving the smallest distance 
        if dist < lowest_dist:
            lowest_dist = dist
            most_likely = name

    return most_likely


print(who_is_it(""" enter the image path here"""))











