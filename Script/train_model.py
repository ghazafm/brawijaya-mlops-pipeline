import os
import argparse
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import logging
import mlflow 


log_dir = os.path.join(os.path.join(os.getcwd()), "Log")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "train.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_data(data_dir):
    logging.info(f"Loading data from {data_dir}...")
    X_train_path = os.path.join(data_dir, "X_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    X_test_path = os.path.join(data_dir, "X_test.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")

    if not os.path.exists(X_train_path):
        raise FileNotFoundError(f"The file {X_train_path} does not exist.")

    if not os.path.exists(X_test_path):
        raise FileNotFoundError(f"The file {X_test_path} does not exist.")
    
    if not os.path.exists(y_train_path):
        raise FileNotFoundError(f"The file {y_train_path} does not exist.")

    if not os.path.exists(y_test_path):
        raise FileNotFoundError(f"The file {y_test_path} does not exist.")
    
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]  # Extract first column if y_train is a DataFrame

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]  # Extract first column if y_train is a DataFrame
        
    logging.info("Data loaded successfully.")
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, model_name="xgboost", params=None):
    with mlflow.start_run():
        logging.info(f"Training the model: {model_name}...")
        mlflow.autolog()
        # Choose model based on input
        if model_name == "xgboost":
            model = XGBClassifier(**(params or {}))
        elif model_name == "lgbm" or model_name == "gbdt":
            model = LGBMClassifier(**(params or {}))
        # elif model_name == 'catboost':
        #     model = CatBoostClassifier(verbose=0, **(params or {}))
        elif model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100, random_state=42, **(params or {})
            )
        elif model_name == "svm":
            model = SVC(**(params or {}))
        elif model_name == "logistic_regression":
            model = LogisticRegression(**(params or {}))
        else:
            raise ValueError(
                f"Model {model_name} is not supported. Choose from 'xgboost', 'catboost', 'lgbm', 'gbdt', 'random_forest', 'svm', or 'logistic_regression'."
            )

        try:
            model.fit(X_train, y_train)
            logging.info(f"Model {model_name} trained successfully.")
            accuracy = evaluate_model(model, X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            if model_name == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            elif model_name == "lgbm":
                mlflow.lightgbm.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise

        return model


def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating the model...")
    y_train_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_train_pred)
    logging.info(f"Training Accuracy: {accuracy:.4f}")
    return accuracy


def save_model(model, model_dir, model_name, timestamp):
    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Combine model_name with timestamp
    model_name_with_time = f"{model_name}_{timestamp}"

    # Use model_name_with_time in the file path
    model_path = os.path.join(model_dir, f"{model_name_with_time}.pkl")

    logging.info(f"Saving model to {model_path}...")

    try:
        # Save the model as a .pkl file
        joblib.dump(model, model_path)
        logging.info(f"Model saved successfully as {model_path}.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

    return model_path


def main(data_dir, model_dir, timestamp, model_name="xgboost", params=None):
    # Load training data
    X_train, X_test, y_train, y_test = load_data(data_dir)

    # Train the model
    model = train_model(X_train,X_test, y_train, y_test, model_name, params)

    # Save the trained model
    save_model(model, model_dir, model_name, timestamp)

    logging.info("Model training and saving completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Directory where the processed data is stored.",
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        required=True,
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        default="xgboost",
        help="Model to train. Options: 'xgboost', 'lgbm', 'random_forest', 'svm', 'logistic_regression'.",
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default=None,
        help="Optional model hyperparameters in JSON format.",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp when Makefile executed.",
    )

    args = parser.parse_args()

    # Convert params from JSON string to Python dictionary, if provided
    params = None
    if args.params:
        import json

        params = json.loads(args.params)

    main(
        args.data_dir, args.model_dir, args.timestamp, args.model_name, params
    )
