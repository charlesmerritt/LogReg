import argparse
import numpy as np

STEP_SIZE = 0.0001
N_ITERS = 200000


def create_feature_matrix(data, num_documents, num_features):
    """Generate a feature matrix from the given data."""
    feature_matrix = {}
    for row in data:
        doc_id, word_id, count = int(row[0]), int(row[1]), row[2]
        if doc_id not in feature_matrix:
            feature_matrix[doc_id] = {}
        feature_matrix[doc_id][word_id] = count
    return feature_matrix


def initialize_weights(num_features):
    """Initialize the weights to zero."""
    return np.zeros(num_features)


def dict_to_numpy_array(feature_dict, num_documents, num_features):
    """Convert a dictionary-based feature matrix to a NumPy array."""
    X = np.zeros((num_documents, num_features))
    for doc_id, features in feature_dict.items():
        for feature_id, count in features.items():
            X[doc_id - 1, feature_id - 1] = count
    return X


def sigmoid(X, weights):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-X.dot(weights)))


def cost_function(hypothesis, y):
    """Compute the logistic regression cost."""
    return (-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)).mean()


def gradient_descent(X, y, weights, alpha, num_iterations):
    """Perform gradient descent to optimize weights."""
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X, weights)
        gradient = (1 / m) * X.T.dot(h - y)
        weights -= alpha * gradient
    return weights


def load_data_and_train(train_data_path, train_label_path):
    """Load training data and labels, train the model, and return the optimized weights."""
    train_data = np.loadtxt(train_data_path)
    train_labels = np.loadtxt(train_label_path)
    num_documents = int(max(train_data[:, 0]))
    num_features = int(max(train_data[:, 1]))
    feature_matrix_numpy = create_feature_matrix(train_data, num_documents, num_features)
    feature_matrix_np = dict_to_numpy_array(feature_matrix_numpy, num_documents, num_features)
    feature_matrix_with_bias = np.hstack((np.ones((num_documents, 1)), feature_matrix_np))
    optimized_weights = gradient_descent(feature_matrix_with_bias, train_labels, np.zeros(num_features + 1), STEP_SIZE,
                                         N_ITERS)
    return optimized_weights, num_features


def make_predictions(test_data_path, optimized_weights, num_features):
    """Load test data, make predictions using the trained model, and print the results."""
    test_data = np.loadtxt(test_data_path)
    num_test_documents = int(max(test_data[:, 0]))
    test_feature_matrix = create_feature_matrix(test_data, num_test_documents, num_features)
    test_feature_matrix_np = dict_to_numpy_array(test_feature_matrix, num_test_documents, num_features)
    test_feature_matrix_with_bias = np.hstack((np.ones((num_test_documents, 1)), test_feature_matrix_np))
    test_predictions = sigmoid(test_feature_matrix_with_bias, optimized_weights)
    for pred in test_predictions:
        print(1 if pred >= 0.5 else 0)


if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description="Homework 1",
                                     epilog="CSCI 4360/6360 Data Science II: Fall 2023",
                                     add_help="How to use",
                                     prog="python homework1.py [train-data] [train-label] [test-data]")
    parser.add_argument("paths", nargs=3)
    args = vars(parser.parse_args())

    optimized_weights, num_features = load_data_and_train(args["paths"][0], args["paths"][1])
    make_predictions(args["paths"][2], optimized_weights, num_features)
