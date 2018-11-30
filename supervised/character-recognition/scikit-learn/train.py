import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load digits training set
digits = datasets.load_digits()

# Split the training set into 75% training and 25% test data
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25, random_state = 0)

# Use logistic regression model
model = LogisticRegression(solver = 'liblinear', multi_class = 'auto')
# Train the model
model.fit(x_train, y_train)

# Save the model
pickle.dump(model, open('model.joblib', 'wb'))
