
## Accuracy graph for each pca value

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the full MNIST dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Select a subset of the training data
train_X = train_X[:60000]
train_y = train_y[:60000]
test_X = test_X[:10000]
test_y = test_y[:10000]

train_X = train_X / 255.0
test_X = test_X / 255.0

# Perform PCA dimensionality reduction
pca_accuracy = []
for n_components in range(1, 30):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(train_X.reshape((60000, 28*28)))
    X_test_pca = pca.transform(test_X.reshape((10000, 28*28)))

    # Train a SVM classifier
    svm = SVC(kernel='rbf', C=1.0, gamma=0.1)
    svm.fit(X_train_pca, train_y)

    # Evaluate the classifier on the testing set
    accuracy = svm.score(X_test_pca, test_y)
    pca_accuracy.append(accuracy)
    print(f"n_components = {n_components}, Accuracy: {accuracy}")

# Plot the results
plt.plot(range(1, 30), pca_accuracy, marker='o')
plt.xlabel('Number of PCA Components')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of PCA Components')
plt.show()