import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Model import KNN

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42)

# ================= PCA A 2 DIMENSIONS =================
# Reduce the data to 2 dimensions using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a scatter plot
plt.figure(figsize=(10, 6))

# Plot the data points, color by the target variable (0 = malignant, 1 = benign)
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='red', label='Malignant', alpha=0.5)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='blue', label='Benign', alpha=0.5)

# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Breast Cancer Dataset')
plt.legend()
plt.show()

# ================ PCA a 3 dimensions ===================
# Reduce the data to 3 dimensions using PCA
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter(X_pca_3d[y == 0, 0], X_pca_3d[y == 0, 1], X_pca_3d[y == 0, 2], c='red', label='Malignant', alpha=0.5)
ax.scatter(X_pca_3d[y == 1, 0], X_pca_3d[y == 1, 1], X_pca_3d[y == 1, 2], c='blue', label='Benign', alpha=0.5)

# Set labels and title
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.title('3D PCA of Breast Cancer Dataset')
plt.legend()
plt.show()

clf = KNN(n_neighbors= 5, metric= 'euclidean', weights= 'distance', algorithm= 'kd_tree' ,n_jobs= -1)
clf.fit(X_train, y_train)
predictions= clf.predict(X_test)

print(y_test)
print(predictions)
score = clf.score(y_test, predictions)
print(score)

