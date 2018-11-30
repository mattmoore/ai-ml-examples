import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
nltk.download('stopwords')
nltk.download('wordnet')
import pickle
from nltk.corpus import stopwords

movie_data = load_files(r'txt_sentoken')
X, y = movie_data.data, movie_data.target

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = pickle.load(open('model.pkl', 'rb'))
prediction = model.predict(X_test)

print("Prediction: ", prediction)
