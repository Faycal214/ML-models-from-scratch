import pandas as pd
from pca import PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import pour la visualisation 3D


data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)  # Convertir en DataFrame
y = data.target  # Cibles

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

pca = PCA()
pca.fit(X_train)
X_train_projected = pca.transform(X_train)

print(f"shape of X : {X_train.shape}")
print(f"shape of x transformed : {X_train_projected.shape}")

# visualization before PCA
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracer les données dans l'espace 3D
scatter = ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], X_train.iloc[:, 2], c=y_train, cmap='viridis')
ax.set_xlabel(X_train.columns[0])
ax.set_ylabel(X_train.columns[1])
ax.set_zlabel(X_train.columns[2])
plt.title("Avant PCA (espace 3D)")

# Ajouter une légende
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)

# Sauvegarder la figure
plt.savefig('iris_3d_visualization.png', dpi=300, bbox_inches='tight')  # dpi pour la résolution
plt.show()

# visualisation after PCA
plt.figure()
plt.scatter(X_train_projected[:, 0], X_train_projected[:, 1], c=y_train, cmap='viridis')
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Après PCA")
plt.savefig('iris_pca_projection.png', dpi=300, bbox_inches='tight')  # dpi pour la résolution
plt.show()