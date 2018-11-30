from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

x = np.array([
  [1, 2],
  [1.5, 1.8],
  [5, 8],
  [8, 8],
  [1, 0.6],
  [9, 11]
])

model = KMeans(n_clusters = 2)
model.fit(x)

centroids = model.cluster_centers_
labels = model.labels_

print("Centroids: ", centroids)
print("Labels: ", labels)

colors = ["g.", "r.", "c.", "b.", "k."]

for i in range(len(x)):
  plt.plot(x[i][0], x[i][1], colors[labels[i]], markersize = 25)

plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 150, linewidths=5)
plt.show()
