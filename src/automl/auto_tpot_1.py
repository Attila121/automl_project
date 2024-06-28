#description: This script trains a TPOT model on multiple datasets and evaluates its performance.   

import pandas as pd
import numpy as np
from tpot import TPOTRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train TPOT on multiple datasets and evaluate performance.')
parser.add_argument('--task', choices=['all', 'y_prop_4_1', 'Bike_Sharing_Demand', 'Brazilian_Houses'], default='all',
                    help='Specify which dataset to train on or use "all" for all datasets.')
parser.add_argument('--seed', type=int, default=42,
                    help='Seed for random number generators.')
parser.add_argument('--output-path', type=str, default='preds.npy',
                    help='Path to save predictions.')
args = parser.parse_args()

# Set random seed for reproducibility
seed = args.seed
np.random.seed(seed)

# Define datasets
datasets = {
    'y_prop_4_1': '361092',
    'Bike_Sharing_Demand': '361099',
    'Brazilian_Houses': '361098'
}

# Filter datasets based on the task argument
if args.task != 'all':
    datasets = {args.task: datasets[args.task]}

# Initialize scores dictionary to store R2 scores for each dataset
scores = {name: [] for name in datasets}

# Parameters for TPOT optimization
tpot_params = {
    'generations': 3,         # Number of generations to run the genetic algorithm
    'population_size': 10,    # Number of pipelines in each generation
    'random_state': seed      # Seed for reproducibility
}

def evaluate_dataset(name, folder, fold, X_train, y_train, X_test, y_test, tpot_params):
    """
    Preprocesses the data, fits the TPOT model, and evaluates it on the test set.

    Parameters:
    name (str): Name of the dataset.
    folder (str): Folder where the dataset is stored.
    fold (int): Current fold number.
    X_train (DataFrame): Training features.
    y_train (Series): Training target.
    X_test (DataFrame): Testing features.
    y_test (Series): Testing target.
    tpot_params (dict): Parameters for TPOTRegressor.

    Returns:
    float: R2 score of the TPOT model on the test set.
    """
    print(f"Loading data for {name}, fold {fold}")

    # Define numerical and categorical columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

    # Define preprocessing pipelines for numerical and categorical features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with median
        ('scaler', StandardScaler())                   # Standardize numerical features
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent
        ('encoder', OneHotEncoder(handle_unknown='ignore'))    # One-hot encode categorical features
    ])

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # Preprocess the data
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    # Initialize and fit TPOT
    tpot = TPOTRegressor(verbosity=2, **tpot_params)
    tpot.fit(X_train_prepared, y_train)

    # Predict and evaluate the model
    y_pred = tpot.predict(X_test_prepared)
    r2 = r2_score(y_test, y_pred)

    print(f"Dataset: {name}, Fold: {fold}, R2 score: {r2}")
    return r2

# Loop through each dataset and each fold
for name, folder in datasets.items():
    for fold in range(1, 11):  # Assuming 10 folds
        # Load data
        X_train = pd.read_parquet(os.path.join(os.path.dirname(__file__), '..', '..', 'data', folder, str(fold), 'X_train.parquet'))
        y_train = pd.read_parquet(os.path.join(os.path.dirname(__file__), '..', '..', 'data', folder, str(fold), 'y_train.parquet')).values.ravel()
        X_test = pd.read_parquet(os.path.join(os.path.dirname(__file__), '..', '..', 'data', folder, str(fold), 'X_test.parquet'))
        y_test = pd.read_parquet(os.path.join(os.path.dirname(__file__), '..', '..', 'data', folder, str(fold), 'y_test.parquet')).values.ravel()

        # Evaluate dataset fold
        r2_score_fold = evaluate_dataset(name, folder, fold, X_train, y_train, X_test, y_test, tpot_params)
        scores[name].append(r2_score_fold)

# Print overall performance
print("\nReference performance:")
for name in scores:
    print(f"{name}: {np.mean(scores[name]):.3f}")

# Plot the R2 scores
fig, ax = plt.subplots()
for name in scores:
    ax.plot(range(1, 11), scores[name], label=name)

ax.set_xlabel('Fold')
ax.set_ylabel('R2 Score')
ax.set_title('R2 Scores for TPOT on Multiple Datasets with Feature Engineering')
ax.legend()
plt.show()

# Save the scores
output_path = args.output_path
np.save(output_path, scores)
