from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest.randomForest import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

estimator = DecisionTree()
estimator.fit(X_train, y_train)
predictions = estimator.predict(X_test)

def accuracy(y_true, y_pred) :
    return np.sum(y_true == y_pred) / len(y_true)

acc = accuracy(y_test, predictions)
print(acc)
