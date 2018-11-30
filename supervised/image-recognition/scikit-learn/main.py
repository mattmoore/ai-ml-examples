# import pandas as pd
# import numpy as np
# import sklearn as sl
# from PIL import Image

# image_files = list(map(lambda file: '../images/' + file, ['apple.jpg', 'orange.jpg', 'banana.jpg']))
# 
# images = []
# for file in image_files:
#     images.append(np.asarray(Image.open(file)))
# 
# print(images[0].data.shape)

from sklearn import tree
features = [[140,1], [130,1], [150,0], [170, 0]]
labels = [0,0,1,1]
target_names = ['Apple', 'orange']
features_name = ['weight', 'texture']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[160,0]]))
