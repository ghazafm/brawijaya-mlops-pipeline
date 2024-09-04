import os
import argparse
import pandas as pd
import joblib
import logging

log_dir = os.path.join(os.path.dirname(os.getcwd()), 'CODE/Log')    
os.makedirs(log_dir, exist_ok=True) 
log_file_path = os.path.join(log_dir, 'predict.log')

def load_model(model_path):
    logging.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
    return model

def load_preprocessor(preprocessor_dir):
    logging.info(f"Loading preprocessor objects from {preprocessor_dir}...")
    
    # Load all preprocessing objects
    scaler = joblib.load(os.path.join(preprocessor_dir, 'scaler.pkl'))
    # label_encoders = joblib.load(os.path.join(preprocessor_dir, 'label_encoders.pkl'))
    imputer = joblib.load(os.path.join(preprocessor_dir, 'imputer.pkl'))
    
    logging.info("Preprocessor objects loaded successfully.")
    return scaler, imputer

def load_new_data(data_dir):
    logging.info(f"Loading new data from {data_dir}...")
    new_data_path = os.path.join(data_dir, 'test.csv')  # Assuming new data is stored in new_data.csv
    new_data = pd.read_csv(new_data_path)
    logging.info("New data loaded successfully.")
    return new_data

def preprocess_data(new_data, scaler, imputer):
    logging.info("Applying imputer to fill missing values...")
    new_data_imputed = pd.DataFrame(imputer.transform(new_data), columns=new_data.columns)
    
    # logging.info("Applying label encoders for categorical columns...")
    # for col, encoder in label_encoders.items():
    #     new_data_imputed[col] = encoder.transform(new_data_imputed[col])
    
    logging.info("Scaling the data...")
    new_data_scaled = pd.DataFrame(scaler.transform(new_data_imputed), columns=new_data_imputed.columns)
    
    return new_data_scaled

def save_predictions(predictions, output_file):
    logging.info(f"Saving predictions to {output_file}...")
    predictions.to_csv(output_file, index=False)
    logging.info("Predictions saved successfully.")

def main(model_path, preprocessor_dir, data_dir, output_file):
    # Load the trained model
    model = load_model(model_path)
    
    # Load preprocessing objects
    scaler, imputer = load_preprocessor(preprocessor_dir)
    
    # Load new data for prediction
    new_data = load_new_data(data_dir)
    
    # Preprocess the new data
    new_data_processed = preprocess_data(new_data, scaler, imputer)
    
    # Make predictions
    logging.info("Making predictions...")
    predictions = model.predict(new_data_processed)
    
    # Save predictions
    save_predictions(pd.DataFrame(predictions, columns=['Prediction']), output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained model and preprocessor.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model.")
    parser.add_argument('--preprocessor', type=str, required=True, help="Directory where the preprocessor objects are stored.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory where the new data is stored.")
    parser.add_argument('--output', type=str, required=True, help="File to save the predictions.")
    
    args = parser.parse_args()
    main(args.model, args.preprocessor, args.data_dir, args.output)
