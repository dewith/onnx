"""
A small script to convert a sklearn model to ONNX.
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.33, random_state=42
    )
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "out/models/iris_clf.joblib")


if __name__ == "__main__":
    main()
