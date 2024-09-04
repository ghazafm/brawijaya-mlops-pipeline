# Paths
DATA_DIR = Data/
MODEL_DIR = Model/model
RESULTS_DIR = Result/scores
TEST_RESULT_DIR = Result/test 
PREPROCESSOR_DIR = Model/preprocessor
NEW_DATA_DIR = Data/
COLUMN_TO_REMOVE = "Cabin"

TIMESTAMP = $(shell date +"%Y%m%d_%H%M%S")
PREDICTION_RESULTS = $(TEST_RESULT_DIR)/predictions_$(TIMESTAMP).csv

LATEST_MODEL = $(shell ls -t $(MODEL_DIR)/*.pkl | head -n 1)

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

# Step 1: Data collection and preparation
.PHONY: data
data:
	@echo "Collecting and preparing data..."
	python scripts/data_preparation.py --data_dir $(DATA_DIR) --output_dir $(PREPROCESSOR_DIR) --target_col target_column --columns_to_remove $(COLUMN_TO_REMOVE)
	@echo "Data preparation completed."

# Step 2: Model training
.PHONY: train
train: data
	@echo "Training the machine learning model..."
	python scripts/train_model.py --data_dir $(DATA_DIR) --model_dir $(MODEL_DIR)
	@echo "Model training completed."

# Step 3: Model evaluation
.PHONY: evaluate
evaluate: train
	@echo "Evaluating the trained model..."
	python scripts/evaluate_model.py --model $(LATEST_MODEL) --results_dir $(RESULTS_DIR)
	@echo "Model evaluation completed."

# Step 4: Model deployment (saving the model)
.PHONY: deploy
deploy: train
	@echo "Deploying the trained model..."
	python scripts/deploy_model.py --model $(LATEST_MODEL) --model_dir $(MODEL_DIR) --metadata_dir $(RESULTS_DIR)
	@echo "Model has been saved and deployed."

# Step 5: Prediction on new data
.PHONY: predict
predict: deploy
	@echo "Running predictions on new data..."
	python scripts/predict_data.py --model $(LATEST_MODEL) --preprocessor $(PREPROCESSOR_DIR) --data_dir $(NEW_DATA_DIR) --output $(PREDICTION_RESULTS)
	@echo "Predictions saved to $(PREDICTION_RESULTS)."

# Clean the data, model, and result directories
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(DATA_DIR)/* $(MODEL_DIR)/* $(RESULTS_DIR)/* $(TEST_RESULT_DIR)/*
	@echo "Clean up completed."
