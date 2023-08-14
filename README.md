# Portable-JSON-Implementation-of-FaceNet-Model

## This project is taken from the online course on Coursera: Deep Learning Specialization.

This repository is a portable implemination of the JSON FaceNet model from the online course Deep Learning Specialization, week 4 of Convolutional Neural Networks. 
The model in the course makes use of Lambda layers, which are not portable since they have deserialization limits, so within the JSON file, I replaced the 
Lambda layers with idle Layer objects (tf.keras.layers.Layer). 

## Accuracy

The model seems to have a 70%-80% accuracy in identifying the person in the image (not that good). So you are wishing to use the JSON model and its weights
for real-world applications, you should add a few more Dense layers and train with a dataset around 6000 - 10000 images, using triplet loss
