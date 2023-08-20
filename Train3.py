## COde with the misclassified numbers

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
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
pca = PCA(n_components=23)
X_train_pca = pca.fit_transform(train_X.reshape((60000, 28*28)))
X_test_pca = pca.transform(test_X.reshape((10000, 28*28)))

# Train a SVM classifier
svm = SVC(kernel='rbf', C=1.0, gamma=0.1)
svm.fit(X_train_pca, train_y)

# Evaluate the classifier on the testing set
y_pred = svm.predict(X_test_pca)
accuracy = svm.score(X_test_pca, test_y)
print("Accuracy:", accuracy)

# Compute precision, recall, and F1 score
print(classification_report(test_y, y_pred))

# Compute confusion matrix
cm = confusion_matrix(test_y, y_pred)

# Visualize misclassified digits
misclassified_idx = np.where(y_pred != test_y)[0]
num_images = 5
fig, axes = plt.subplots(num_images, num_images, figsize=(10,10))
for i in range(num_images):
    for j in range(num_images):
        idx = misclassified_idx[i*num_images+j]
        axes[i,j].imshow(test_X[idx], cmap='gray')
        axes[i,j].set_title(f"True: {test_y[idx]}\nPredicted: {y_pred[idx]}")
plt.tight_layout()
plt.show()

# Visualize confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=[str(i) for i in range(10)],
       yticklabels=[str(i) for i in range(10)],
       title='Confusion matrix',
       ylabel='True label',
       xlabel='Predicted label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
fmt = '.2f'  # Use 2 decimal places
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.show()