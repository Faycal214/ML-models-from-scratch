import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from randomForest import RandomForest

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 1234)

def accuracy(y_true, y_pred) :
    np.sum(y_true == y_pred) / len(y_true)

rf = RandomForest()
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print(accuracy(y_test, predictions))


