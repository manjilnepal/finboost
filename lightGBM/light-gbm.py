## --- IMPORT LIBRARIES --- ##
import pandas as pd
import numpy as np
import os
import shutil
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import sys

sys.path.append('../utilities')

from preprocess import preprocess

LR = 0.001
ROUNDS = 500


## --- DEFINE PATH --- ##
PARTICIPANT_DATA_PATH = '/home/dgxuser40/manjil/finsurv/participant_data'
SUBMISSION_DIR = f'{LR}_{ROUNDS}_boost_lgbm'
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

        # LightGBM parameters optimized for survival analysis
        params = {
            'objective': 'regression_l2',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'max_depth': 3,
            'learning_rate': LR,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'verbose': -1,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'min_gain_to_split': 0.1
        }

        print("Training model...")

        # Format labels for survival analysis
        # Use timeDiff as label
        y_train_label = y_train['timeDiff'].values
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train_label)

        # Train the model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=ROUNDS,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(period=0)]
        )

        print("Model trained successfully.")

        # --- generate and save predictions ---
        print(f"Generating predictions for {dataset_path}...")
        predictions = model.predict(X_test_processed)
        
        # Negate predictions as required by competition
        predictions_negated = -predictions
        
        # save predictions to a CSV file
        prediction_filename = dataset_path.replace(os.sep, '_') + '.csv'
        prediction_save_path = os.path.join(SUBMISSION_DIR, prediction_filename)
        pd.DataFrame(predictions_negated).to_csv(prediction_save_path, header=False, index=False)
        print(f"  -> Predictions saved to {prediction_save_path}")
        print(f"  -> Prediction range: [{predictions_negated.min():.6f}, {predictions_negated.max():.6f}]")
        
    except Exception as e:
        print(f"\nERROR: The model for {dataset_path} failed to train.")
        print(f"Details: {e}")

print("\n\nAll prediction files have been generated.")


## --- CREATE SUBMISSION FOLDER --- ##
output_zip_filename = f'{LR}_{ROUNDS}_boost_lgbm'
shutil.make_archive(output_zip_filename, 'zip', SUBMISSION_DIR)
print(f"Successfully created '{output_zip_filename}.zip' from the '{SUBMISSION_DIR}' directory.")
print("You can now upload this file to the CodaBench competition.")