######################################################
#  Student Name: Alex Tamai
#  Student ID: W30591647
#  Course Code: CSCI 460 -- Fall 2025
#  Assignment Due Date: December 1, 2025
#  GitHub Link: https://github.com/tamaia2/CSCI460F2025_W30591647_Programing_Assignemnt2
######################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#############################
# 1) Get the Brain Tumor Data
#############################

# variables
directory = '/data/csci460/BTD'
category = ['yes', 'no']
images_paths = []
image_labels = []

# reading through the yes/no
for cat in category:
    path = os.path.join(directory, cat)
    files = os.listdir(path)
    for filename in files:
        full = os.path.join(path, filename)
        images_paths.append(full)
        image_labels.append(cat)

df = pd.DataFrame({'image_paths':images_paths,
                'image_labels':image_labels})

########################
# 2) Build the CNN model
########################

# split data into training and testing
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['image_labels']
)

gen = ImageDataGenerator()

# training data generator
train_gen = gen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='image_label',
    target_size=(224, 224),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=16
)

# testing data generator
test_gen = gen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='image_label',
    target_size=(224, 224),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=16
)

# model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid')
])

# compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
