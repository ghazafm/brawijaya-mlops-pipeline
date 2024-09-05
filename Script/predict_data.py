import os
import argparse
import pandas as pd
import joblib
import logging

log_dir = os.path.join(os.path.dirname(os.getcwd()), "CODE/Log")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "predict.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_model(model_path):
    logging.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
    return model


def load_preprocessor(preprocessor_dir,timestamp):
    logging.info(f"Loading preprocessor objects from {preprocessor_dir}...")

    # Load all preprocessing objects
    scaler = joblib.load(os.path.join(preprocessor_dir, f"scaler_{timestamp}.pkl"))
    label_encoders = joblib.load(os.path.join(preprocessor_dir, f"label_encoders_{timestamp}.pkl"))
    imputer = joblib.load(os.path.join(preprocessor_dir, f"imputer_{timestamp}.pkl"))
    removed_cols = joblib.load(os.path.join(preprocessor_dir, f"removed_cols_{timestamp}.pkl"))

    logging.info("Preprocessor objects loaded successfully.")
    return scaler, imputer, label_encoders, removed_cols


def load_new_data(data_dir, id_col):
    logging.info(f"Loading new data from {data_dir}...")
    new_data_path = os.path.join(data_dir, "test.csv")
    new_data = pd.read_csv(new_data_path)

    ids = new_data[id_col]
    logging.info(f"New data and ID column loaded successfully.")

    return new_data, ids


def preprocess_data(new_data, scaler, imputer, encoder, removed_cols, id_col):
    logging.info("Removing specified columns, including the ID column...")

    columns_to_remove = [col for col in removed_cols if col != id_col]
    new_data = new_data.drop(columns=columns_to_remove, errors="ignore")

    logging.info("Applying imputer to fill missing values...")
    new_data_imputed = pd.DataFrame(
        imputer.transform(new_data), columns=new_data.columns
    )

    logging.info("Applying label encoders for categorical columns...")
    for col, enc in encoder.items():
        new_data_imputed[col] = enc.transform(new_data_imputed[col].astype(str))

    logging.info("Scaling the data...")
    new_data_scaled = pd.DataFrame(
        scaler.transform(new_data_imputed), columns=new_data_imputed.columns
    )

    if id_col in new_data_scaled.columns:
        new_data_scaled = new_data_scaled.drop(columns=[id_col])

    return new_data_scaled



def save_predictions(predictions, ids, predictions_dir, model_name, timestamp):
    os.makedirs(predictions_dir, exist_ok=True)
    prediction_name_with_time = f"{model_name}_{timestamp}"

    predictions = pd.DataFrame({"ID": ids, "Prediction": predictions})

    logging.info(f"Saving predictions to {prediction_name_with_time}...")
    predictions.to_csv(prediction_name_with_time, index=False)
    logging.info("Predictions saved successfully.")


def main(model_path, id_col, preprocessor_dir, data_dir, timestamp):
    # Load the trained model
    model = load_model(model_path)

    # Load preprocessing objects
    scaler, imputer, encoder, removed_cols = load_preprocessor(preprocessor_dir,timestamp)

    # Load new data for prediction
    new_data, ids = load_new_data(data_dir, id_col)

    # Preprocess the new data
    new_data_processed = preprocess_data(
        new_data, scaler, imputer, encoder, removed_cols, id_col
    )

    # Make predictions
    logging.info("Making predictions...")
    predictions = model.predict(new_data_processed)

    # Save predictions
    save_predictions(pd.DataFrame(predictions, columns=["Prediction"]), timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make predictions using a trained model and preprocessor."
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to the trained model."
    )
    parser.add_argument(
        "-i",
        "--id_col",
        type=str,
        required=True,
        help="Column Id where identified each row.",
    )
    parser.add_argument(
        "-p",
        "--preprocessor",
        type=str,
        required=True,
        help="Directory where the preprocessor objects are stored.",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Directory where the new data is stored.",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp when Makefile executed.",
    )

    args = parser.parse_args()
    main(model_path=args.model, id_col=args.id_col, preprocessor_dir=args.preprocessor, data_dir=args.data_dir, timestamp=args.timestamp)
