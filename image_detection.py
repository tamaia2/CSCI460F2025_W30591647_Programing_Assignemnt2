######################################################
#  Student Name: Alex Tamai
#  Student ID: W30591647
#  Course Code: CSCI 460 -- Fall 2025
#  Assignment Due Date: December 1, 2025
#  GitHub Link: https://github.com/tamaia2/CSCI460F2025_W30591647_Programing_Assignemnt2
######################################################

import os, sys
import tensorflow as tf
import numpy as np
import sklearn.model_selection as sklearnmodels

#############################
# Function for reading images
#############################
def readImageDirectory(basedir, classDict, imageSize=None, quiet=True, testRatio=0.33):
  # Initialize the X tensor and the Y vector raw structures
  imagesX = []
  imagesY = []

  for classKey in classDict:
    dirName = os.path.join(basedir, classKey)
    for filename in os.listdir(dirName):
        # Filename name is the base name + class name + image file name
        fn = os.path.join(dirName, filename)

        # Load the image
        rawImage = tf.keras.preprocessing.image.load_img(fn, target_size=imageSize )
	# Scale image to all three color channel
        image = tf.keras.preprocessing.image.img_to_array(rawImage)/255.0

        # Grow the image tensor - increase the size or the number of dimensions (rank) of the tensor
        imagesX.append(image)
	# The class vector by 1 entry
        imagesY.append(classDict[classKey])

  # Return these as a tensor and a numpy vector
  trainX, testX, trainY, testY = sklearnmodels.train_test_split(imagesX, imagesY, test_size=testRatio)
  return tf.convert_to_tensor(trainX), tf.convert_to_tensor(testX), np.array(trainY, dtype="float32"), np.array(testY, dtype="float32")

#############################
# 1) Get the Brain Tumor Data
#############################

# Grab all the data from: /data/csci460/BTD
trainX, testX, trainY, testY = readImageDirectory("/data/csci460/BTD", {"yes":1, "no":0}, (224,224), False)

# Setting up the Y labels to be compatible with a "hot-ones" representation
trainY = tf.keras.utils.to_categorical(trainY)
testY  = tf.keras.utils.to_categorical(testY)
