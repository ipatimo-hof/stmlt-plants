

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load the model
m = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1')

# Load the image and resize it
image = Image.open('C:/Users/ptimofeev/Pictures/plants/f2.jpg')
image = image.resize((224, 224))

# Convert the image to a numpy array and add a batch dimension
image = np.array(image)
image = np.expand_dims(image, axis=0)
# Convert the image to a float tensor and normalize it
image = tf.cast(image, tf.float32) / 255.0
# Pass the image to the model and get the output tensor
output = m(image)

# Find the index of the category with the highest logit value and print it
index = np.argmax(output)
print(index)