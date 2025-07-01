# Teacheing-Machine-to-discrimination-between-two-types-of-cars
Car Classification Script
This Python script uses a pre-trained deep learning model to classify images of Dodge and Corvette cars. Here's how it works:

Loads the Model:

Uses a saved Keras model (keras_Model.h5) trained specifically to recognize Dodge vs. Corvette cars.

Reads class labels from labels.txt (0 Dodge, 1 Corvette).

Processes the Input Image:

Takes a car image ("dodge_charger.jpg") and resizes/crops it to 224x224 pixels.

Normalizes pixel values to match the model's requirements.

Predicts the Car Model:

Runs the image through the model to predict whether it's a Dodge or Corvette.

Outputs the predicted class and confidence score (0.98 for 98% confidence).

Example Output:
text
Class: Dodge  
Confidence Score: 0.98  
Key Features:
Specialized for Cars: Optimized to distinguish between Dodge (e.g., Challenger, Charger) and Corvette models.

Easy to Customize: Replace keras_Model.h5 with a model trained on other car brands (e.g., Ford vs. Chevrolet).

Dependencies: Keras, TensorFlow, Pillow, NumPy.

How to Use:
Replace <IMAGE_PATH> with your car image (e.g., "corvette.jpg").

Ensure labels.txt contains:

0 Dodge  
1 Corvette  
Run the script to see the classification!

Code:

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("<IMAGE_PATH>").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
