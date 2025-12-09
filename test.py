import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

data_dir = '/data/csci460/BTD'
groups = ['yes', 'no']
file_paths = []
labels = []

for group in groups:
    fold_path = os.path.join(data_dir, group)
    files = os.listdir(fold_path)
    for file in files:
        file_path = os.path.join(fold_path, file)
        file_paths.append(file_path)
        labels.append(group)

df=pd.DataFrame({'file_paths':file_paths,
                'labels':labels})

print(df.sample(10))

print(df['labels'].value_counts())

train_df,test_df=train_test_split(df,test_size=0.2,random_state=42,stratify=df['labels'])

gen=ImageDataGenerator()
train_gen=gen.flow_from_dataframe(train_df, x_col='file_paths', y_col='labels',
                                    target_size=(224, 224), color_mode='grayscale',
                                    class_mode='binary', batch_size=16)

test_gen=gen.flow_from_dataframe(test_df, x_col='file_paths', y_col='labels',
                                    target_size=(224, 224), color_mode='grayscale',
                                    class_mode='binary', batch_size=16)

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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

traning = model.fit(train_gen, epochs=25, validation_data=test_gen)

model.evaluate(train_gen)

model.evaluate(test_gen)
plt.figure(figsize=(7,5))
plt.plot(traning.history['accuracy'], label='Train Accuracy')
plt.plot(traning.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.savefig('accuracy.png')
plt.show()

plt.figure(figsize=(7,5))
plt.plot(traning.history['loss'], label='Train Loss')
plt.plot(traning.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.savefig('loss.png')
plt.show()

import random
from tensorflow.keras.preprocessing import image

def show_prediction(folder, filename):
    img_path = os.path.join(data_dir, folder, filename)
    img = image.load_img(img_path, target_size=(224,224), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result = "Tumor"
        color = "red"
    else:
        result = "No Tumor"
        color = "green"

    plt.imshow(np.array(img).squeeze(), cmap='gray')
    plt.title(f"Actual: {folder.upper()}\nPredicted: {result}", color=color)
    plt.axis('off')


yes_images = random.sample(os.listdir(os.path.join(data_dir, 'yes')), 3)
no_images  = random.sample(os.listdir(os.path.join(data_dir, 'no')), 3)

plt.figure(figsize=(12, 8))

for i, img_name in enumerate(yes_images):
    plt.subplot(2, 3, i + 1)
    show_prediction('yes', img_name)

for i, img_name in enumerate(no_images):
    plt.subplot(2, 3, i + 4)
    show_prediction('no', img_name)

plt.tight_layout()
plt.show()
