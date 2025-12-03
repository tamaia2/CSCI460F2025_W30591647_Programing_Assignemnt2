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
# 1) Get the Brain Tumor Data
#############################

# Grab all the data from that directory
# From: /data/csci460/BTD
trainX, testX, trainY, testY = readImageDirectory("/data/csci460/BTD", {"yes":1, "no":0}, (224,224), False)

# Let's setup the Y labels to be compatible with a "hot-ones" representation
trainY = tf.keras.utils.to_categorical(trainY)
testY  = tf.keras.utils.to_categorical(testY)

#############################
# Function for reading images
#############################
def readImageDirectory(basedir, classDict, imageSize=None, quiet=True, testRatio=0.33):
  """
  Take a base directory and a dictionary of class values,
  to into that directory and read all the images there to
  build an image data set.  It is assumed that different
  classes are in different subdirectories with that class
  name.  Return the tensor with the images, as well as the
  label Y vector.  If a tuple (x,y) imageSize is given,
  then enforce all the images are of a specific size.
  """
  # Initialize the X tensor and the Y vector raw structures
  imagesX = []
  imagesY = []

  for classKey in classDict:
    dirName = os.path.join(basedir, classKey)
    for filename in os.listdir(dirName):
        # Filename name is the base name + class name + image file name
        fn = os.path.join(dirName, filename)

        # If we want to, we can print the images file names as we read them
        if not quiet:
          print(fn)

        # Load the image, then make sure it is scaled all all three color channels
        rawImage = tf.keras.preprocessing.image.load_img(fn, target_size=imageSize )
        image = tf.keras.preprocessing.image.img_to_array(rawImage)/255.0

        # Grow the image tensor and the class vector by 1 entry
        imagesX.append(image)
        imagesY.append(classDict[classKey])

  # Return these as a tensor and a numpy vector
  trainX, testX, trainY, testY = sklearnmodels.train_test_split(imagesX, imagesY, test_size=testRatio)
  return tf.convert_to_tensor(trainX), tf.convert_to_tensor(testX), np.array(trainY, dtype="float32"), np.array(testY, dtype="float32")
