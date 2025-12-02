# Brain Tumor Detection using CNN

## Project Description
This project implements a deep learning system to classify brain MRI images for detecting the presence or absence of brain tumors. Using a Convolutional Neural Network (CNN) built with TensorFlow, it processes and classifies images from a balanced dataset of 1500 positive (tumor) and 1500 negative MRI scans.

## Dataset
- The MRI images are organized into two folders: `/data/csci460/BTD/yes` (tumor present) and `/data/csci460/BTD/no` (tumor absent).
- Each class contains 1500 images.
- Images are directly loaded and preprocessed from these directories without copying to save space.

## Features
- Image preprocessing includes resizing to 224x224 pixels and normalization.
- Dataset split into training (60%), validation (20%), and testing (20%) subsets.
- CNN architecture with convolutional, max pooling, dense layers, and softmax output for binary classification.
- Training with categorical cross-entropy loss and stochastic gradient descent.
- Evaluation includes overall accuracy and class-wise performance to detect bias between tumor-present and tumor-absent classes.

## Usage Instructions

### Environment Setup
source /data/shared-venvs/tensorflow-standard/bin/activate

### Running the Training Script
python images_detection.py

### Outputs
- Trained CNN model.
- Performance metrics printed for training, validation, and separate class-wise test accuracies.

## Report
The project report (3 pages max) includes description of design choices, parameter tuning, evaluation methodology, and results discussion.

---
