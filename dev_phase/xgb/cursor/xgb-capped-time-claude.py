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
ROUNDS = 500  # Increase from 500

## --- DEFINE PATH --- ##
PARTICIPANT_DATA_PATH = '/home/dgxuser40/manjil/finsurv/participant_data'
SUBMISSION_DIR = f'claude_timecap_{LR}_{ROUNDS}_boost_xgb'
os.makedirs(SUBMISSION_DIR, exist_ok=True)

def cap_timeDiff_outliers(y_train, cap_percentile=99):
    """Cap extreme timeDiff outliers while preserving censoring validity"""
    y_train = y_train.copy()  # Fix the warning
    
    cap_value = y_train['timeDiff'].quantile(cap_percentile / 100)
    
    # Identify values beyond cap
    beyond_cap = y_train['timeDiff'] > cap_value
    
    # Cap the timeDiff
    y_train['timeDiff_capped'] = np.clip(y_train['timeDiff'], 0, cap_value)
    
    # For EVENTS (status=1) beyond cap, convert to censored
    y_train['status_adjusted'] = y_train['status'].copy()
    y_train.loc[beyond_cap & (y_train['status'] == 1), 'status_adjusted'] = 0
    
    print(f"Cap value: {cap_value:,.0f}, Events converted to censored: {((beyond_cap) & (y_train['status'] == 1)).sum()}")
    
    return y_train

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

        # XGBoost parameters optimized for C-index
        # BEST PARAMETERS for C-index > 0.84
        params = {
            'objective': 'survival:cox',
            'eval_metric': 'cox-nloglik',
            'max_depth': 6,           # Increase from 5
            'eta': LR,              # Lower learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 4,    # Decrease from 5
            'lambda': 1.0,            # Lower regularization
            'alpha': 0.1,             # Lower regularization
            'gamma': 0.1,             # Add this
            'seed': 42
        }


        ##-- feature selection phase ---##
        print("Training model...")

        y_train = cap_timeDiff_outliers(y_train, cap_percentile=99)  

        y_train_xgb = np.where(
            y_train['status_adjusted'] == 1,  # Changed from 'status' to 'status_adjusted'
            y_train['timeDiff_capped'],
            -y_train['timeDiff_capped']
        )

        # create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train_xgb)
        dtest = xgb.DMatrix(X_test_processed)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=ROUNDS,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=100,
            verbose_eval=100
        )

        # After training your model, add this:
        print("=== DIAGNOSTIC INFO ===")
        print(f"TimeDiff stats: Min={y_train['timeDiff'].min()}, Max={y_train['timeDiff'].max()}")
        print(f"TimeDiff median: {y_train['timeDiff'].median()}")
        print(f"Censored events: {(y_train['status'] == 0).sum()}")
        print(f"Uncensored events: {(y_train['status'] == 1).sum()}")

        # Get feature importance
        importance = model.get_score(importance_type='gain')
        importance_df = pd.DataFrame(list(importance.items()), columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=False)
        print(f"Top 10 features: {importance_df.head(10)['feature'].tolist()}")

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


## --- CREATE SUMBISSION FOLDER --- ##
output_zip_filename = f'claude_timecap_{LR}_{ROUNDS}_boost_xgb'
shutil.make_archive(output_zip_filename, 'zip', SUBMISSION_DIR)
print(f"Successfully created '{output_zip_filename}.zip' from the '{SUBMISSION_DIR}' directory.")
print("You can now upload this file to the CodaBench competition.")