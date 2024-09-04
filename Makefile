# Paths
DATA_DIR = Data/
MODEL_DIR = Model/model
SCORE_RESULTS_DIR = Result/scores
TEST_RESULT_DIR = Result/test 
METADATA_RESULT_DIR = Result/test 
PREPROCESSOR_DIR = Model/preprocessor
NEW_DATA_DIR = Data/
DEPLOYED_MODEL_FILE = $(MODEL_DIR)/deploy_model_path.txt  
COLUMN_TO_REMOVE = Cabin 
TARGET_COL = Survived 
RANDOM_STATE = 42

TIMESTAMP = $(shell date +"%Y%m%d_%H%M%S")
PREDICTION_RESULTS = $(TEST_RESULT_DIR)/predictions_$(TIMESTAMP).csv

# Default target
.PHONY: all
all: help

# Display help for the Makefile
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make data      : Run data collection and preparation steps."
	@echo "  make train     : Train the machine learning model."
	@echo "  make evaluate  : Evaluate the trained model."
	@echo "  make deploy    : Save and deploy the trained model."
	@echo "  make predict   : Run predictions on new data."
	@echo "  make clean     : Clean data, model, and result directories."
	@echo "  make clean_models     : Clean only the trained models."
	@echo "  make clean_results    : Clean only the prediction and score results."
	@echo "  make clean_preprocessor    : Clean only the preprocessor objects."

# Step 1: Data collection and preparation
.PHONY: data
data:
	@echo "Collecting and preparing data..."
	python Script/data_preparation.py --data_dir $(DATA_DIR) --output_dir $(PREPROCESSOR_DIR) --target_col $(TARGET_COL) --random_state $(RANDOM_STATE) --columns_to_remove $(COLUMN_TO_REMOVE)
	@echo "Data preparation completed."

# Step 2: Model training
.PHONY: train
train: data
	@echo "Training the machine learning model..."
	python Script/train_model.py --data_dir $(DATA_DIR) --model_dir $(MODEL_DIR)
	@echo "Model training completed."

# Step 3: Model evaluation
.PHONY: evaluate
evaluate: train
	@echo "Evaluating the trained model..."
	python Script/evaluate_model.py --model $(LATEST_MODEL) --results_dir $(SCORE_RESULTS_DIR)
	@echo "Model evaluation completed."

# Step 4: Model deployment (saving the model)
.PHONY: deploy
deploy: train
	@echo "Deploying the trained model..."
	python Script/deploy_model.py --model_path $(LATEST_MODEL) --model_dir $(MODEL_DIR) --metadata_dir $(METADATA_RESULT_DIR) > $(DEPLOYED_MODEL_FILE)
	@echo "Model has been saved and deployed. Model path stored in $(DEPLOYED_MODEL_FILE)."

# Step 5: Prediction on new data using the deployed model
.PHONY: predict
predict: deploy
	@echo "Running predictions on new data using the deployed model..."
	DEPLOYED_MODEL=$(shell cat $(DEPLOYED_MODEL_FILE))
	python Script/predict_data.py --model $$DEPLOYED_MODEL --preprocessor $(PREPROCESSOR_DIR) --data_dir $(NEW_DATA_DIR) --output $(PREDICTION_RESULTS)
	@echo "Predictions saved to $(PREDICTION_RESULTS)."

# Clean all data, models, and results
.PHONY: clean
clean: clean_models clean_results clean_preprocessor
	@echo "Complete cleanup completed."

# Clean only models
.PHONY: clean_models
clean_models:
	@echo "Cleaning up models..."
	rm -rf $(MODEL_DIR)/*.pkl $(MODEL_DIR)/$(DEPLOYED_MODEL_FILE)
	@echo "Models cleaned."

# Clean only results (predictions and scores)
.PHONY: clean_results
clean_results:
	@echo "Cleaning up results (scores and predictions)..."
	rm -rf $(SCORE_RESULTS_DIR)/* $(TEST_RESULT_DIR)/*
	@echo "Results cleaned."

# Clean only preprocessor objects
.PHONY: clean_preprocessor
clean_preprocessor:
	@echo "Cleaning up preprocessor objects..."
	rm -rf $(PREPROCESSOR_DIR)/*
	@echo "Preprocessor objects cleaned."
