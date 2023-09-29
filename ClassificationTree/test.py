from sklearn.datasets import make_blobs, make_classification
import numpy as np
import matplotlib.pyplot as plt
from classification_tree import ClassificationTree

# test on blobs
blobs = make_blobs(100, 2)

dataset = np.hstack((blobs[0], blobs[1].reshape((-1, 1))))

plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2])
plt.show()

classifier = ClassificationTree(2, 2)

classifier.fit(dataset[:, :-1], dataset[:, -1].reshape((-1, 1)))

preds = classifier.predict(dataset[:, :-1])

plt.scatter(dataset[:, 0], dataset[:, 1], c=preds)
plt.show()

# test on worse blobs
cla = make_classification(
    n_samples=100,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
)


dataset = np.hstack((cla[0], cla[1].reshape((-1, 1))))

classifier = ClassificationTree(2, 2)

classifier.fit(dataset[:, :-1], dataset[:, -1].reshape((-1, 1)))

preds = classifier.predict(dataset[:, :-1])

np.abs(preds - cla[1]).sum()
