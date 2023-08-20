from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from joblib import dump

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
pca = PCA(n_components=25)
X_train_pca = pca.fit_transform(train_X.reshape((60000, 28*28)))
X_test_pca = pca.transform(test_X.reshape((10000, 28*28)))

# Train a SVM classifier
svm = SVC(kernel='rbf', C=1.0, gamma=0.1)
svm.fit(X_train_pca, train_y)

# Evaluate the classifier on the testing set
accuracy = svm.score(X_test_pca, test_y)
print("Accuracy:", accuracy)

# Calculate the confusion matrix
y_pred = svm.predict(X_test_pca)
cm = confusion_matrix(test_y, y_pred)
print("Confusion Matrix:\n", cm)

# Save the trained model to a file
dump(svm, 'svm_model.joblib')
dump(pca,'pca_model.joblib')


