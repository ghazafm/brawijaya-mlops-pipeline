import os
import argparse
import joblib
import json
import logging
from datetime import datetime

log_dir = os.path.join(os.path.join(os.getcwd()), "Log")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "deploy.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_model(model_path):
    logging.info(f"Loading model from {model_path}...")

    if not os.path.exists(model_path):
        logging.error(f"Model file {model_path} not found.")
        raise FileNotFoundError(f"The model file {model_path} does not exist.")

    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")

    return model


def extract_model_metadata(model):
    model_metadata = {}

    # Identify model type based on its class name
    model_type = type(model).__name__
    model_metadata["model_type"] = model_type

    # Extract model parameters (if applicable)
    if hasattr(model, "get_params"):
        model_metadata["parameters"] = model.get_params()

    # For tree-based models, extract additional details
    if model_type in ["RandomForestClassifier", "XGBClassifier", "LGBMClassifier"]:
        if hasattr(model, "n_estimators"):
            model_metadata["n_estimators"] = model.n_estimators

    return model_metadata, model_type


def save_model(model, model_dir, model_name, timestamp):
    os.makedirs(model_dir, exist_ok=True)

    # Create the full model path with model name, "deployed", and timestamp
    model_path = os.path.join(model_dir, f"{model_name}_deployed_{timestamp}.pkl")

    logging.info(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    logging.info(f"Model saved successfully as {model_path}.")

    return model_path


def save_metadata(model_name, timestamp, model_path, metadata_dir, metadata):
    os.makedirs(metadata_dir, exist_ok=True)

    # Create metadata file path with model name and timestamp
    metadata_file_path = os.path.join(
        metadata_dir, f"{model_name}_metadata_{timestamp}.json"
    )

    # Add standard metadata dynamically from the input metadata
    metadata["model_name"] = model_name
    metadata["model_path"] = model_path
    metadata["deployed_timestamp"] = timestamp

    logging.info(f"Saving model metadata to {metadata_file_path}...")
    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info("Model metadata saved successfully.")


def deploy_model(model_path, model_dir, metadata_dir, additional_metadata, timestamp):
    # Load the trained model
    model = load_model(model_path)

    # Extract dynamic metadata and model name from the model
    model_metadata, model_name = extract_model_metadata(model)

    # Merge additional metadata passed from the command line (if any)
    if additional_metadata:
        model_metadata.update(additional_metadata)

    # Save the model with a timestamp and model name
    saved_model_path = save_model(model, model_dir, model_name, timestamp)

    # Save metadata with model name and timestamp
    save_metadata(model_name, timestamp, saved_model_path, metadata_dir, model_metadata)

    # Output the path of the deployed model
    print(saved_model_path)

    logging.info("Model deployment completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy a trained machine learning model."
    )

    # Arguments
    parser.add_argument(
        "-p",
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        required=True,
        help="Directory to save the deployed model.",
    )
    parser.add_argument(
        "-md",
        "--metadata_dir",
        type=str,
        required=True,
        help="Directory to save model metadata.",
    )
    parser.add_argument(
        "-ma",
        "--metadata",
        type=str,
        help="Optional additional model metadata in JSON format.",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp when Makefile executed.",
    )

    args = parser.parse_args()

    # Load additional metadata if provided
    additional_metadata = {}
    if args.metadata:
        try:
            additional_metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding metadata: {e}")
            raise ValueError("Invalid metadata JSON format.")

    # Deploy the model with automatically extracted model name and additional metadata
    deploy_model(
        args.model_path, args.model_dir, args.metadata_dir, additional_metadata, timestamp=args.timestamp
    )
