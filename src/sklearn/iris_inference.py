"""
A small script to load a onnx model and run inference.
"""

import logging

import joblib
import numpy as np
import onnx
import onnxruntime

from sklearn.metrics import accuracy_score


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("* Starting the inference on the iris dataset")

    # Load the iris dataset
    logger.info("Loading the iris dataset")
    X_test = np.load("out/data/X_test.npy")
    y_test = np.load("out/data/y_test.npy")

    # Load the onnx model
    logger.info("Loading the onnx model")
    clf_onnx = onnx.load("out/models/iris_clf.onnx")
    logger.info("Loading the pickle model")
    clf = joblib.load("out/models/iris_clf.pkl")

    # Run inference with python
    logger.info("Running inference with python")
    y_pred = clf.predict(X_test)
    logger.info("-> Predictions: %s", y_pred[:10])
    logger.info("-> Accuracy: %.4f", accuracy_score(y_test, y_pred))

    # Run inference with onnx
    logger.info("Running inference with onnx")
    session = onnxruntime.InferenceSession(clf_onnx.SerializeToString())
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    y_pred_onnx = session.run(
        output_names=[label_name],
        input_feed={input_name: X_test.astype(np.float32)},
    )[0]
    logger.info("-> Predictions: %s", y_pred_onnx[:10])
    logger.info("-> Accuracy: %.4f", accuracy_score(y_test, y_pred_onnx))

    logger.info("* Done")


if __name__ == "__main__":
    main()
