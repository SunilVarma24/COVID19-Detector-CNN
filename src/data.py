# src/data.py
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_images_labels(dataset_path, img_size=128):
    images = []
    labels = []
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                # Load the image using OpenCV and convert it to grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # Resize the image to a larger common size (e.g., 128x128 pixels)
                image = cv2.resize(image, (img_size, img_size))
                # Convert grayscale to RGB by replicating the gray channel to three channels
                image = np.stack((image,)*3, axis=-1)
                # Normalize pixel values to the range [0, 1]
                image = image / 255.0
                # Append the image and label to lists
                images.append(image)
                labels.append(class_name)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def encode_labels(labels):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    labels_encoded = labels_encoded.reshape(-1, 1)
    labels_onehot = onehot_encoder.fit_transform(labels_encoded)
    return labels_onehot, label_encoder