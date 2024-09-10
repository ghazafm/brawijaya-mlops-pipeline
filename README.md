# Machine Learning Operations Project

This project demonstrates a complete machine learning workflow, including data preparation, model training, evaluation, and prediction. The project structure is organized to separate different stages and files for better management and reproducibility.

## Project Structure

```
├── Data
│   ├── clean            # Cleaned data ready for training
│   └── raw              # Raw, unprocessed data
├── Log                  # Logs generated during various processes
├── Model
│   ├── metadata         # Metadata for the models
│   ├── model            # Trained models
│   └── preprocessor     # Preprocessing objects (e.g., scalers)
├── myenv                # Virtual environment (Conda or venv)
├── Notebook             # Jupyter notebooks for analysis and experiments
├── Result
│   ├── predict          # Predictions on new data
│   └── scores           # Model evaluation scores
├── Script               # Python scripts for different stages of ML lifecycle
│   ├── data_preparation.py    # Prepares the data
│   ├── train_model.py         # Trains the model
│   ├── evaluate_model.py      # Evaluates the model
│   ├── deploy_model.py        # Deploys the trained model
│   └── predict_data.py        # Predicts new data using the deployed model
├── .gitignore            # Files and directories to be ignored by Git
├── environment.yml       # Conda environment file
├── Makefile              # Automates common tasks
├── README.md             # Project documentation (this file)
├── requirements.txt      # Python dependencies
└── timestamp.txt         # Timestamp for different operations
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ghazafm/MLOps.git
cd MLOps
```

### 2. Create a Virtual Environment

#### Conda (Recommended)
```bash
make conda_env
```

#### Virtual Environment (venv)
```bash
make venv_env
```

### 3. Activate the Virtual Environment
- Conda:
  ```bash
  conda activate ./myenv
  ```
- Virtualenv:
  ```bash
  source myenv/bin/activate
  ```

## Running the Project

### 1. Data Preparation
To clean and prepare the data:
```bash
make data
```

### 2. Model Training
To train the model:
```bash
make train
```

### 3. Model Evaluation
To evaluate the model:
```bash
make evaluate
```

### 4. Model Prediction
To run predictions on new data:
```bash
make predict
```

## Dependencies

All required dependencies are listed in:
- `requirements.txt` (for venv)
- `environment.yml` (for Conda)

## License

This project is licensed under the MIT License.
