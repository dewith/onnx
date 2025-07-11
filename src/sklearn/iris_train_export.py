"""
A small script to train a sklearn model and export it to ONNX.
"""

import logging
import os

import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("* Starting the training on the iris dataset")

    # Create a directory for the models
    os.makedirs("out/models", exist_ok=True)
    os.makedirs("out/data", exist_ok=True)

    # Load the iris dataset
    logger.info("Loading the iris dataset")
    iris = load_iris()
    logger.info("Splitting the dataset into train and test")
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.33, random_state=42
    )

    # Save the data
    logger.info("Saving the data")
    np.save("out/data/X_train.npy", X_train)
    np.save("out/data/X_test.npy", X_test)
    np.save("out/data/y_train.npy", y_train)
    np.save("out/data/y_test.npy", y_test)

    # Train the model
    logger.info("Training the model")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    logger.info("Evaluating the model")
    y_pred = clf.predict(X_test)
    accuracy_train = accuracy_score(y_train, clf.predict(X_train))
    accuracy_test = accuracy_score(y_test, y_pred)
    logger.info("Accuracy on train set: %.4f", accuracy_train)
    logger.info("Accuracy on test set: %.4f", accuracy_test)

    # Convert the model to ONNX
    logger.info("Converting the model to ONNX")
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    clf_onnx = convert_sklearn(clf, initial_types=initial_type)

    # Save the model
    logger.info("Saving the model as an ONNX file")
    with open("out/models/iris_clf.onnx", "wb") as f:
        f.write(clf_onnx.SerializeToString())

    # Save the model
    logger.info("Saving the model as a pickle file")
    joblib.dump(clf, "out/models/iris_clf.pkl")

    logger.info("* Done")


if __name__ == "__main__":
    main()
