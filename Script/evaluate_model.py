import datetime
import os
import argparse
import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
import logging

log_dir = os.path.join(os.path.join(os.getcwd()), "Log")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "evaluation.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_data(data_dir):
    logging.info(f"Loading test data from {data_dir}...")
    X_test_path = os.path.join(data_dir, "X_test.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")

    if not os.path.exists(X_test_path):
        raise FileNotFoundError(f"The file {X_test_path} does not exist.")

    if not os.path.exists(y_test_path):
        raise FileNotFoundError(f"The file {y_test_path} does not exist.")

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]  # Convert y_test to Series if it's a DataFrame

    logging.info("Test data loaded successfully.")
    return X_test, y_test


def load_model(model_path):
    logging.info(f"Loading model from {model_path}...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")

    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
    return model


def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating the model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-score: {f1:.4f}")

    # Generate a detailed classification report
    report = classification_report(y_test, y_pred)
    logging.info("Classification Report:\n" + report)

    return accuracy, precision, recall, f1, report


def save_evaluation_results(
    results_dir, model, accuracy, precision, recall, f1, report, timestamp
):
    os.makedirs(results_dir, exist_ok=True)

    result_name_time = f"{model}_{timestamp}"

    results_path = os.path.join(results_dir, f"{result_name_time}.txt")

    logging.info(f"Saving evaluation results to {results_path}...")

    with open(results_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    logging.info("Evaluation results saved successfully.")


def main(model_path, data_dir, results_dir, timestamp):
    # Load test data
    X_test, y_test = load_data(data_dir)

    # Load the trained model
    model = load_model(model_path)
    model_type = type(model).__name__

    # Evaluate the model
    accuracy, precision, recall, f1, report = evaluate_model(model, X_test, y_test)

    # Save evaluation results
    save_evaluation_results(
        results_dir, model_type, accuracy, precision, recall, f1, report, timestamp
    )

    logging.info("Model evaluation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained machine learning model."
    )

    # Arguments
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model file.",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Directory where the test data is stored.",
        default="Data/",
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results.",
        default="Result/scores",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp when Makefile executed.",
    )

    args = parser.parse_args()

    # Execute the main function
    main(model_path=args.model_path, data_dir=args.data_dir, results_dir=args.results_dir, timestamp=args.timestamp)
