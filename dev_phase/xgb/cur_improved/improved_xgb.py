## --- IMPROVED XGBOOST WITH TRANSITION-SPECIFIC OPTIMIZATION --- ##
import pandas as pd
import numpy as np
import os
import shutil
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import sys

sys.path.append('.')

from improved_preprocess import preprocess

# Transition-specific hyperparameters
TRANSITION_PARAMS = {
    # Strong performers - can use more aggressive parameters
    ('Withdraw', 'Liquidated'): {
        'max_depth': 8, 'eta': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.9,
        'min_child_weight': 1, 'lambda': 1.0, 'alpha': 0.1, 'gamma': 0.0
    },
    ('Withdraw', 'Repay'): {
        'max_depth': 7, 'eta': 0.025, 'subsample': 0.85, 'colsample_bytree': 0.85,
        'min_child_weight': 2, 'lambda': 1.5, 'alpha': 0.2, 'gamma': 0.0
    },
    ('Withdraw', 'Borrow'): {
        'max_depth': 7, 'eta': 0.025, 'subsample': 0.85, 'colsample_bytree': 0.85,
        'min_child_weight': 2, 'lambda': 1.5, 'alpha': 0.2, 'gamma': 0.0
    },
    ('Deposit', 'Liquidated'): {
        'max_depth': 7, 'eta': 0.025, 'subsample': 0.85, 'colsample_bytree': 0.85,
        'min_child_weight': 2, 'lambda': 1.5, 'alpha': 0.2, 'gamma': 0.0
    },
    ('Deposit', 'Repay'): {
        'max_depth': 6, 'eta': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 3, 'lambda': 2.0, 'alpha': 0.3, 'gamma': 0.1
    },
    ('Deposit', 'Borrow'): {
        'max_depth': 6, 'eta': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 3, 'lambda': 2.0, 'alpha': 0.3, 'gamma': 0.1
    },
    ('Deposit', 'Withdraw'): {
        'max_depth': 6, 'eta': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 3, 'lambda': 2.0, 'alpha': 0.3, 'gamma': 0.1
    },
    # Weak performers - need more conservative parameters
    ('Repay', 'Borrow'): {
        'max_depth': 5, 'eta': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'min_child_weight': 5, 'lambda': 3.0, 'alpha': 0.5, 'gamma': 0.2
    },
    ('Repay', 'Withdraw'): {
        'max_depth': 4, 'eta': 0.06, 'subsample': 0.6, 'colsample_bytree': 0.6,
        'min_child_weight': 7, 'lambda': 4.0, 'alpha': 0.7, 'gamma': 0.3
    },
    ('Repay', 'Liquidated'): {
        'max_depth': 4, 'eta': 0.06, 'subsample': 0.6, 'colsample_bytree': 0.6,
        'min_child_weight': 7, 'lambda': 4.0, 'alpha': 0.7, 'gamma': 0.3
    },
    ('Repay', 'Deposit'): {
        'max_depth': 5, 'eta': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'min_child_weight': 5, 'lambda': 3.0, 'alpha': 0.5, 'gamma': 0.2
    },
    ('Borrow', 'Deposit'): {
        'max_depth': 5, 'eta': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'min_child_weight': 5, 'lambda': 3.0, 'alpha': 0.5, 'gamma': 0.2
    },
    ('Borrow', 'Withdraw'): {
        'max_depth': 4, 'eta': 0.06, 'subsample': 0.6, 'colsample_bytree': 0.6,
        'min_child_weight': 7, 'lambda': 4.0, 'alpha': 0.7, 'gamma': 0.3
    },
    ('Borrow', 'Liquidated'): {
        'max_depth': 4, 'eta': 0.06, 'subsample': 0.6, 'colsample_bytree': 0.6,
        'min_child_weight': 7, 'lambda': 4.0, 'alpha': 0.7, 'gamma': 0.3
    },
    ('Borrow', 'Repay'): {
        'max_depth': 5, 'eta': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'min_child_weight': 5, 'lambda': 3.0, 'alpha': 0.5, 'gamma': 0.2
    },
    ('Withdraw', 'Deposit'): {
        'max_depth': 6, 'eta': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 3, 'lambda': 2.0, 'alpha': 0.3, 'gamma': 0.1
    }
}

# Transition-specific rounds
TRANSITION_ROUNDS = {
    # Strong performers - fewer rounds to avoid overfitting
    ('Withdraw', 'Liquidated'): 600,
    ('Withdraw', 'Repay'): 700,
    ('Withdraw', 'Borrow'): 700,
    ('Deposit', 'Liquidated'): 700,
    ('Deposit', 'Repay'): 800,
    ('Deposit', 'Borrow'): 800,
    ('Deposit', 'Withdraw'): 800,
    ('Withdraw', 'Deposit'): 800,
    # Weak performers - more rounds for better learning
    ('Repay', 'Borrow'): 1000,
    ('Repay', 'Withdraw'): 1200,
    ('Repay', 'Liquidated'): 1200,
    ('Repay', 'Deposit'): 1000,
    ('Borrow', 'Deposit'): 1000,
    ('Borrow', 'Withdraw'): 1200,
    ('Borrow', 'Liquidated'): 1200,
    ('Borrow', 'Repay'): 1000,
}

## --- DEFINE PATH --- ##
PARTICIPANT_DATA_PATH = '/home/dgxuser40/manjil/finsurv/participant_data'
SUBMISSION_DIR = f'improved_transition_specific_xgb'
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

## --- START TRAINING PROCESS --- ##
for index_event, outcome_event in event_pairs:
    print(f"\n{'='*60}")
    print(f"Processing and Predicting for: {index_event} -> {outcome_event}")
    print(f"{'='*60}")
    
    dataset_path = os.path.join(index_event, outcome_event)
    
    try:
        train_df = pd.read_csv(os.path.join(PARTICIPANT_DATA_PATH, dataset_path, 'train.csv'))
        test_features_df = pd.read_csv(os.path.join(PARTICIPANT_DATA_PATH, dataset_path, 'test_features.csv'))
    except FileNotFoundError as e:
        print(f"Data not found for {dataset_path}. Skipping.")
        continue
        
    # Use transition-specific preprocessing
    X_train, y_train, X_test_processed = preprocess(
        train_df, test_features_df, index_event, outcome_event
    )

    try:
        # Get transition-specific parameters
        base_params = TRANSITION_PARAMS.get((index_event, outcome_event), {
            'max_depth': 6, 'eta': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'min_child_weight': 3, 'lambda': 2.0, 'alpha': 0.3, 'gamma': 0.1
        })
        
        rounds = TRANSITION_ROUNDS.get((index_event, outcome_event), 800)
        
        # XGBoost parameters optimized for C-index
        params = {
            'objective': 'survival:cox',
            'eval_metric': 'cox-nloglik',
            'seed': 42,
            **base_params
        }

        print(f"Using parameters: {params}")
        print(f"Training for {rounds} rounds...")

        ##-- feature selection phase ---##
        print("Training model...")

        y_train_xgb = np.where(
        y_train['status'] == 1,
        np.log(y_train['timeDiff'] + 1),  # log transform for events
        -np.log(y_train['timeDiff'] + 1)  # log transform for censored
        )

        # create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train_xgb)
        dtest = xgb.DMatrix(X_test_processed)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        print("  - Model trained successfully.")

        # --- generate and save predictions ---
        print(f"Generating predictions for {dataset_path}...")
        predictions = -model.predict(dtest)
        
        # save predictions to a CSV file
        prediction_filename = dataset_path.replace(os.sep, '_') + '.csv'
        prediction_save_path = os.path.join(SUBMISSION_DIR, prediction_filename)
        pd.DataFrame(predictions).to_csv(prediction_save_path, header=False, index=False)
        print(f"  -> Predictions saved to {prediction_save_path}")
        
    except Exception as e:
        print(f"\nERROR: The model for {dataset_path} failed to train.")
        print(f"Details: {e}")

print("\n\nAll prediction files have been generated.")

## --- CREATE SUBMISSION FOLDER --- ##
output_zip_filename = f'improved_transition_specific_xgb'
shutil.make_archive(output_zip_filename, 'zip', SUBMISSION_DIR)
print(f"Successfully created '{output_zip_filename}.zip' from the '{SUBMISSION_DIR}' directory.")
print("You can now upload this file to the CodaBench competition.")
