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

sys.path.append('../utilities')

from preprocess import preprocess

LR = 0.05
ROUNDS = 500

## --- DEFINE PATH --- ##
PARTICIPANT_DATA_PATH = '/home/dgxuser40/manjil/finsurv/participant_data'
SUBMISSION_DIR = f'0.84_{LR}_{ROUNDS}_boost_xgb'
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

        # XGBoost parameters for survival analysis
        params = {
            'objective': 'survival:cox',
            'eval_metric': 'cox-nloglik',
            'max_depth': 5,
            'eta': LR,  # learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'lambda': 3.0,  # L2 regularization
            'alpha': 0.5,   # L1 regularization
            'seed': 42
        }


        ##-- feature selection phase ---##
        print("Training model for feature selection...")

        y_train_xgb = np.where(
            y_train['status'] == 1,
            y_train['timeDiff'],  # positive for events
            -y_train['timeDiff']  # negative for censored
        )

        # create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train_xgb)
        dtest = xgb.DMatrix(X_test_processed)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=ROUNDS,
            verbose_eval=False
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


## --- CREATE SUMBISSION FOLDER --- ##
output_zip_filename = f'0.84_{LR}_{ROUNDS}_boost_xgb'
shutil.make_archive(output_zip_filename, 'zip', SUBMISSION_DIR)
print(f"Successfully created '{output_zip_filename}.zip' from the '{SUBMISSION_DIR}' directory.")
print("You can now upload this file to the CodaBench competition.")