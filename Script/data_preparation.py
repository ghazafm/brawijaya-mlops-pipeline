import os
import argparse
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import logging

log_dir = os.path.join(os.path.join(os.getcwd()), "Log")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "preparation.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Function to load data
def load_data(file_path):
    if not os.path.exists(file_path):
        logging.error(f"The file {file_path} does not exist.")
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    logging.info(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)


# Function to encode categorical columns
def encode(df):
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns
    logging.info(f"Encoding categorical columns: {categorical_cols.tolist()}")
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


# Function to handle missing values using KNNImputer
def impute(df, n_neighbors=5):
    logging.info(f"Imputing missing values using KNN with {n_neighbors} neighbors...")
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    logging.info(f"Missing values imputed.")
    return df_imputed, imputer


# Function to scale numerical features
def scale(X):
    logging.info(f"Scaling numerical features...")
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    logging.info(f"Numerical columns scaled: {numerical_cols.tolist()}")
    return X, scaler


# Function to save preprocessing objects
def save_preprocessing_objects(
    scaler, label_encoders, imputer, removed_cols, output_dir, timestamp
):
    logging.info(f"Saving preprocessing objects to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, f"scaler_{timestamp}.pkl"))
    joblib.dump(
        label_encoders, os.path.join(output_dir, f"label_encoders_{timestamp}.pkl")
    )
    joblib.dump(imputer, os.path.join(output_dir, f"imputer_{timestamp}.pkl"))
    joblib.dump(removed_cols, os.path.join(output_dir, f"removed_cols_{timestamp}.pkl"))
    logging.info("Preprocessing objects saved successfully.")


# Clean and preprocess the data
def clean_data(df, columns_to_remove=None):
    logging.info("Cleaning data...")

    # Drop specified columns
    if columns_to_remove:
        logging.info(f"Removing columns: {columns_to_remove}")
        df = df.drop(columns=columns_to_remove)

    # Drop Duplicates
    df = df.drop_duplicates()
    logging.info("Duplicates dropped.")

    # Encode categorical columns
    df, label_encoders = encode(df)

    # Impute missing values
    df, imputer = impute(df, n_neighbors=5)

    logging.info("Data cleaning completed.")
    return df, label_encoders, imputer


# Split data into training and testing sets
def split_data(df, target, test_size=0.2, random_state=42):
    logging.info(
        f"Splitting data into train and test sets with test_size={test_size} and random_state={random_state}..."
    )

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logging.info("Data split completed.")
    return X_train, X_test, y_train, y_test


# Apply SMOTE for oversampling the minority class
def balancing(X_train, y_train, random_state=42):
    logging.info(
        f"Balancing the training data using SMOTE with random_state={random_state}..."
    )
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logging.info("Training data balanced.")
    return X_train_resampled, y_train_resampled


# Save processed data
def save_data(X_train, X_test, y_train, y_test, data_dir):
    logging.info(f"Saving processed data to {data_dir}...")

    os.makedirs(data_dir, exist_ok=True)

    X_train.to_csv(os.path.join(data_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(data_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(data_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(data_dir, "y_test.csv"), index=False)

    logging.info("Processed data saved successfully.")


# Main function
def main(data_dir,data_new_dir, output_dir, target_col, random_state, columns_to_remove, timestamp):
    # Load raw data
    raw_data_path = os.path.join(data_dir, "train.csv")
    df = load_data(raw_data_path)

    # Clean the data
    df_cleaned, label_encoders, imputer = clean_data(df, columns_to_remove)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(
        df_cleaned, target_col, random_state=random_state
    )

    # Scale features
    X_train, scaler = scale(X_train)
    X_test, _ = scale(X_test)

    # Balance the training data using SMOTE
    X_train_balanced, y_train_balanced = balancing(
        X_train, y_train, random_state=random_state
    )

    # Save the processed data
    save_data(X_train_balanced, X_test, y_train_balanced, y_test, data_new_dir)

    # Save preprocessing objects for future use (for prediction)
    save_preprocessing_objects(
        scaler, label_encoders, imputer, columns_to_remove, output_dir, timestamp
    )

    logging.info("Data preparation and preprocessing completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and clean data for machine learning."
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Directory where the data is stored and processed.",
    )
    parser.add_argument(
        "-dn",
        "--data_new",
        type=str,
        required=True,
        help="Directory where the clean data is stored.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save preprocessing objects.",
    )
    parser.add_argument(
        "-ta", "--target_col", type=str, required=True, help="Target column name."
    )
    parser.add_argument(
        "-rs",
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    parser.add_argument(
        "-r",
        "--columns_to_remove",
        nargs="+",
        help="List of columns to remove from the dataset.",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp when Makefile executed.",
    )

    args = parser.parse_args()

    main(
        args.data_dir,
        args.data_new,
        args.output_dir,
        args.target_col,
        args.random_state,
        args.columns_to_remove,
        args.timestamp,
    )
