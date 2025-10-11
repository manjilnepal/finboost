## --- IMPORT LIBRARIES --- ##
import pandas as pd
import numpy as np
import os
import shutil
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import sys
from lifelines import WeibullAFTFitter

sys.path.append('../../utilities')

from preprocess import preprocess

LR = 0.05
ROUNDS = 500

## --- DEFINE PATH --- ##
PARTICIPANT_DATA_PATH = '/home/dgxuser40/manjil/finsurv/participant_data'
SUBMISSION_DIR = f'cur_ensemble_{LR}_{ROUNDS}_boost_xgb'
os.makedirs(SUBMISSION_DIR, exist_ok=True)

## --- DEFINE ALL 16 EVENT PAIRS --- ##
index_events = ["Borrow", "Deposit", "Repay", "Withdraw"]
outcome_events = index_events + ["Liquidated"]
event_pairs = []
for index_event in index_events:
    for outcome_event in outcome_events:
        if index_event == outcome_event:
            continue
        event_pairs.append((index_event, outcome_event))

## --- ENSEMBLE MODEL TRAINING --- ##
def train_ensemble_models(X_train, y_train, X_test):
    """
    Train multiple XGBoost models with different parameters and return ensemble predictions
    """
    
    # Model 1: Your current best parameters (0.84 C-index)
    params1 = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': 5,
        'eta': LR,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'lambda': 3.0,
        'alpha': 0.5,
        'seed': 42
    }
    
    # Model 2: Slightly different parameters
    params2 = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': 6,        # Different depth
        'eta': 0.03,           # Different learning rate
        'subsample': 0.85,     # Different subsample
        'colsample_bytree': 0.75,
        'min_child_weight': 4,
        'lambda': 2.0,
        'alpha': 0.3,
        'seed': 123
    }
    
    # Model 3: More aggressive parameters
    params3 = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': 7,
        'eta': 0.08,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 3,
        'lambda': 1.0,
        'alpha': 0.1,
        'seed': 456
    }
    
    # Model 4: Conservative parameters
    params4 = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': 4,
        'eta': 0.02,
        'subsample': 0.75,
        'colsample_bytree': 0.85,
        'min_child_weight': 6,
        'lambda': 4.0,
        'alpha': 0.7,
        'seed': 789
    }
    
    # Model 5: Balanced parameters
    params5 = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': 5,
        'eta': 0.06,
        'subsample': 0.82,
        'colsample_bytree': 0.82,
        'min_child_weight': 4,
        'lambda': 2.5,
        'alpha': 0.4,
        'seed': 101112
    }
    
    models = []
    predictions_list = []
    
    # Train each model
    for i, params in enumerate([params1, params2, params3, params4, params5], 1):
        print(f"Training Model {i}...")
        
        # Prepare target for XGBoost
        y_train_xgb = np.where(
            y_train['status'] == 1,
            y_train['timeDiff'],  # positive for events
            -y_train['timeDiff']  # negative for censored
        )
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train_xgb)
        dtest = xgb.DMatrix(X_test)
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=ROUNDS,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Get predictions
        predictions = -model.predict(dtest)
        predictions_list.append(predictions)
        models.append(model)
        
        print(f"Model {i} trained successfully.")
    
    # Ensemble predictions (weighted average)
    # Give more weight to your best performing model (Model 1)
    weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Weighted towards Model 1
    
    ensemble_predictions = np.average(predictions_list, axis=0, weights=weights)
    
    print(f"Ensemble of {len(models)} models created.")
    
    return ensemble_predictions, models

## --- START TRAINING PROCESS --- ##
for index_event, outcome_event in event_pairs:
    print(f"\n{'='*50}")
    print(f"Processing and Predicting for: {index_event} -> {outcome_event}")
    print(f"{'='*50}")
    
    dataset_path = os.path.join(index_event, outcome_event)
    
    try:
        train_df = pd.read_csv(os.path.join(PARTICIPANT_DATA_PATH, dataset_path, 'train.csv'))
        test_features_df = pd.read_csv(os.path.join(PARTICIPANT_DATA_PATH, dataset_path, 'test_features.csv'))
    except FileNotFoundError as e:
        print(f"Data not found for {dataset_path}. Skipping.")
        continue
        
    X_train, y_train, X_test_processed = preprocess(train_df, test_features_df)

    try:
        print("Training ensemble models...")
        
        # Train ensemble and get predictions
        ensemble_predictions, models = train_ensemble_models(X_train, y_train, X_test_processed)
        
        print("  - Ensemble models trained successfully.")

        # --- generate and save predictions ---
        print(f"Generating ensemble predictions for {dataset_path}...")
        
        # save predictions to a CSV file
        prediction_filename = dataset_path.replace(os.sep, '_') + '.csv'
        prediction_save_path = os.path.join(SUBMISSION_DIR, prediction_filename)
        pd.DataFrame(ensemble_predictions).to_csv(prediction_save_path, header=False, index=False)
        print(f"  -> Ensemble predictions saved to {prediction_save_path}")
        
    except Exception as e:
        print(f"\nERROR: The ensemble for {dataset_path} failed to train.")
        print(f"Details: {e}")

print("\n\nAll ensemble prediction files have been generated.")

## --- CREATE SUBMISSION FOLDER --- ##
output_zip_filename = f'ensemble_hp_{LR}_{ROUNDS}_boost_xgb'
shutil.make_archive(output_zip_filename, 'zip', SUBMISSION_DIR)
print(f"Successfully created '{output_zip_filename}.zip' from the '{SUBMISSION_DIR}' directory.")
print("You can now upload this file to the CodaBench competition.")
