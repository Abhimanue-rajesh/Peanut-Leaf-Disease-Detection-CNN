import os
import cv2
import numpy as np
import tensorflow as tf
import keras

# Load model
model = keras.saving.load_model(os.path.join("model", "model_6.h5"))

# Image path
image_path = "input/image/path/here"

# Read image
img = cv2.imread(image_path)

# Resize image
resize = tf.image.resize(img, (256, 256))
np.expand_dims(resize, 0).shape

# Predict with model
prediction = model.predict(np.expand_dims(resize / 255, 0))
print("Model prediction:", prediction)

# Convert numpy array to string
prediction_str = str(prediction)

# Format the string
formatted_prediction = prediction_str.replace("[[", "").replace("]]", "")

# Convert string to list
prediction_list = list(formatted_prediction.split(" "))

# Find the max value from the list
max_value = max(prediction_list)
max_value_index = prediction_list.index(max_value)

print("Max value:", max_value)
print("Type of max value:", type(max_value))
print("Index of max value:", max_value_index)

# Interpret prediction
if max_value_index == 0:
    print("Not Leaf")
elif max_value_index == 1:
    print("Early Spot")
elif max_value_index == 2:
    print("Late Spot")
elif max_value_index == 3:
    print("Rust")
elif max_value_index == 4:
    print("Normal")

# Convert to percentage
converted_base = float("{:.10f}".format(float(max_value)))
in_percentage = format(converted_base, ".2%")
print("Converted base:", converted_base)
print("In percentage:", in_percentage)
