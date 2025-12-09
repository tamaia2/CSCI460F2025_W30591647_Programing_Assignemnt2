######################################################
#  Student Name: Alex Tamai
#  Student ID: W30591647
#  Course Code: CSCI 460 -- Fall 2025
#  Assignment Due Date: December 9, 2025
######################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# directory for images
directory = '/data/csci460/BTD'

category = ['yes', 'no']
image_paths = []
cat_labels = []

# getting all paths and labels
for cat in category:
    class_dir = os.path.join(directory, cat)
    class_files = os.listdir(class_dir)

    for filename in class_files:
        full = os.path.join(class_dir, filename)
        image_paths.append(full)
        cat_labels.append(cat)

df = pd.DataFrame({
    'image_path': image_paths,
    'cat': cat_labels
})

# split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['cat']
)
# second split
valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df['cat']
)

# generators
data_gen = ImageDataGenerator()
train_generator = data_gen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='cat',
    target_size=(224, 224),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=16
)
test_generator = data_gen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='cat',
    target_size=(224, 224),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=16
)
validation_generator  = data_gen.flow_from_dataframe(
    valid_df,
    x_col='image_path',
    y_col='cat',
    target_size=(224, 224),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=16
)

###########
# CNN model
###########
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

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# train
training_history = model.fit(
    train_generator,
    epochs=25,
    validation_data=test_generator
)

# evaluate
model.evaluate(train_generator)
model.evaluate(test_generator)
model.evaluate(validation_generator)

# plot to show accuracy
plt.figure(figsize=(7, 5))
plt.plot(training_history.history['accuracy'], label='Train Accuracy')
plt.plot(training_history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.savefig('accuracy.png')
plt.show()

# plot to show loss
plt.figure(figsize=(7, 5))
plt.plot(training_history.history['loss'], label='Train Loss')
plt.plot(training_history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.savefig('loss.png')
plt.show()

# Prediction Visualization
import random
from tensorflow.keras.preprocessing import image

def visualize_prediction(cat, filename):
    img_path = os.path.join(directory, cat, filename)
    img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        predicted_label = "Tumor"
        title_color = "red"
    else:
        predicted_label = "No Tumor"
        title_color = "green"

    plt.imshow(np.array(img).squeeze(), cmap='gray')
    plt.title(f"Actual: {cat.upper()}\nPredicted: {predicted_label}", color=title_color)
    plt.axis('off')

# samples of yes and no
tumor_yes = random.sample(os.listdir(os.path.join(directory, 'yes')), 3)
tumor_no = random.sample(os.listdir(os.path.join(directory, 'no')), 3)

plt.figure(figsize=(12, 8))

for i, filename in enumerate(tumor_yes):
    plt.subplot(2, 3, i + 1)
    visualize_prediction('yes', filename)

for i, filename in enumerate(tumor_no):
    plt.subplot(2, 3, i + 4)
    visualize_prediction('no', filename)

plt.tight_layout()
plt.savefig('prediction.png')
plt.show()
