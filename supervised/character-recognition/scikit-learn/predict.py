import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys

# Load the model
model = pickle.load(open('model.joblib', 'rb'))

# Load the data we want to predict using the model
with open(sys.argv[1]) as data_file:
  # Load the data file as a json object, then convert that to a numpy array so we can use it with scikit-learn
  prediction_data = np.asarray(json.load(data_file))
label_names = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# Load the digits
digits = datasets.load_digits()

# Predict image using trained model
prediction = model.predict(prediction_data.reshape(1,-1))

# Print the label name for the image we've predicted
print("Prediction: It's the number %s" % label_names[prediction[0] - 1])


## Printing the image
## We don't have to do this, but to show that the image prediction is correct, we can display the image itself.

# The image data is in a 1-dimensional array for use with the model.
# To draw it with matplotlib we have to reshape the image from a 1-dimensional array to a 2-dimensional array
image_to_draw = prediction_data.reshape(8, -1)
# Set up matplotlib to draw the image
plt.figure(1, figsize=(3, 3))
# Tell matplotlib the reshaped image data to draw
plt.imshow(image_to_draw, cmap = plt.cm.gray_r, interpolation='nearest')
# Tell matplotlib to draw the image
plt.show()
