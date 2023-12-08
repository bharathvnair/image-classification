import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Function to extract HOG features from an image
def extract_features_hog(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 128))  # Resize the image to a standard size for HOG
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    features = features.flatten()  # Flatten the features to use as a feature vector
    return features

# Path to the folder containing celebrity images
data_dir = r"D:\New folder\cropped"

celebrity_features = {}
for celebrity_folder in os.listdir(data_dir):
    celebrity_path = os.path.join(data_dir, celebrity_folder)
    if os.path.isdir(celebrity_path):
        celebrity_features[celebrity_folder] = []
        for img_file in os.listdir(celebrity_path):
            img_path = os.path.join(celebrity_path, img_file)
            features = extract_features_hog(img_path)
            celebrity_features[celebrity_folder].append(features)
len(celebrity_features)

import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
# Convert string labels to integer labels
label_encoder = LabelEncoder()
y = []
for label, features in celebrity_features.items():
    y.extend([label] * len(features))

y_encoded = label_encoder.fit_transform(y)

# Flatten the features for each celebrity
X = []
for features in celebrity_features.values():
    X.extend(features)

X = np.array(X)
y_encoded = np.array(y_encoded)

# Function to create a simple fully connected neural network
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Create the CNN model
num_classes = len(set(y))
input_shape = X.shape[1]

# Assuming each image's feature length is stored in input_shape_length
input_shape_length = X.shape[1]
input_shape = (input_shape_length,)


model = create_model(input_shape=input_shape, num_classes=num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_encoded, epochs=25, batch_size=32, validation_split=0.2)

model.save("celebrity_classification_model.h5")

import cv2
from keras.models import load_model

# Load the saved model
loaded_model = load_model("celebrity_classification_model.h5")

# Function to predict the celebrity from an input image path
def predict_celebrity(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 128))  # Resize the image to match the input size used for training
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    features = features.flatten()  # Flatten the features to use as input for prediction

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(np.array([features]))

    # Get the predicted celebrity label (inverse transform the label encoding)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    return predicted_label

# Example usage:
image_path = r"D:\New folder\cropped\roger_federer\roger_federer18.png"
predicted_celebrity = predict_celebrity(image_path)
print("Predicted celebrity:", predicted_celebrity)