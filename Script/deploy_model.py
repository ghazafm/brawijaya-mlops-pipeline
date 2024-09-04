import os
import argparse
import joblib
import json
import logging
from datetime import datetime

log_dir = os.path.join(os.path.dirname(os.getcwd()), 'Log')  
os.makedirs(log_dir, exist_ok=True) 
log_file_path = os.path.join(log_dir, 'deploy.log')

def load_model(model_path):
    logging.info(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        logging.error(f"Model file {model_path} not found.")
        raise FileNotFoundError(f"The model file {model_path} does not exist.")
    
    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
    
    return model

def save_model(model, model_dir, model_name):
    os.makedirs(model_dir, exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the full model path with timestamp
    model_path = os.path.join(model_dir, f"{model_name}_{timestamp}.pkl")
    
    logging.info(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    logging.info(f"Model saved successfully as {model_path}.")
    
    return model_path

def save_metadata(model_path, metadata_dir, metadata):
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Create metadata file path
    metadata_file_path = os.path.join(metadata_dir, 'model_metadata.json')
    
    # Add model path to the metadata
    metadata['model_path'] = model_path
    
    logging.info(f"Saving model metadata to {metadata_file_path}...")
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logging.info("Model metadata saved successfully.")

def deploy_model(model_path, model_dir, metadata_dir, metadata):
    # Load the trained model
    model = load_model(model_path)
    
    # Save the model with a timestamp
    saved_model_path = save_model(model, model_dir, "deployed_model")
    
    # Save metadata
    save_metadata(saved_model_path, metadata_dir, metadata)
    
    logging.info("Model deployment completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy a trained machine learning model.")
    
    # Arguments
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file.")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory to save the deployed model.")
    parser.add_argument('--metadata_dir', type=str, required=True, help="Directory to save model metadata.")
    parser.add_argument('--metadata', type=str, help="Optional model metadata in JSON format.")
    
    args = parser.parse_args()
    
    # Load metadata if provided
    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding metadata: {e}")
            raise ValueError("Invalid metadata JSON format.")
    
    # Deploy the model
    deploy_model(args.model_path, args.model_dir, args.metadata_dir, metadata)
