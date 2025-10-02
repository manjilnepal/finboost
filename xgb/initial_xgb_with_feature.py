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

## --- DFINE PATH --- ##
PARTICIPANT_DATA_PATH = '/home/dgxuser40/manjil/finsurv/participant_data'
SUBMISSION_DIR = 'xgb_with_50_feature'
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
    # scale features to prevent numerical instability
    # scaler = StandardScaler()
    # X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    # X_test_processed = pd.DataFrame(scaler.transform(X_test_processed), columns=X_test_processed.columns)

    try:

        # XGBoost parameters for survival analysis
        params = {
            'objective': 'survival:cox',
            'eval_metric': 'cox-nloglik',
            'max_depth': 3,
            'eta': 0.01,  # learning rate; changed from 0.1 to 0.01
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'lambda': 5.0,  # L2 regularization; changed from 1.0 to 5.0
            'alpha': 1.0,   # L1 regularization; changed from 0.0 to 1.0
            'seed': 42
        }

        ##-- feature selection phase ---##
        print("Training model for feature selection...")

        #y_train_xgb = y_train['timeDiff'].values 

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
            num_boost_round=100,
            verbose_eval=False
        )

        importance = model.get_score(importance_type='gain')  # 'weight', 'gain', 'cover'
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        TOP_N = 60
        top_features = sorted_features[:TOP_N]
        selected_features = [feature for feature, score in top_features]
        print(f"{len(selected_features)} features selected successfully.")
        
        X_train_new = X_train[selected_features]
        X_test_processed_new = X_test_processed[selected_features]
        y_train_new = y_train.copy()

        ##-- train with selected features ---##
        print("Training model with the selected features...")

        y_train_xgb_new = np.where(
            y_train_new['status'] == 1,
            y_train_new['timeDiff'],  # positive for events
            -y_train_new['timeDiff']  # negative for censored
        )
        
        # Create DMatrix for XGBoost
        dtrain_new = xgb.DMatrix(X_train_new, label=y_train_xgb_new)
        dtest_new = xgb.DMatrix(X_test_processed_new)
        
        print(f"Training XGBoost survival:cox model for {dataset_path}...")
        model_new = xgb.train(
            params,
            dtrain_new,
            num_boost_round=150,
            verbose_eval=False
        )
        print("  - Model trained successfully.")

        # --- generate and save predictions ---
        print(f"Generating predictions for {dataset_path}...")
        # predict risk scores (higher score = higher risk)
        predictions = -model_new.predict(dtest_new)
        
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
output_zip_filename = 'xgb_with_50_feature'
shutil.make_archive(output_zip_filename, 'zip', SUBMISSION_DIR)
print(f"Successfully created '{output_zip_filename}.zip' from the '{SUBMISSION_DIR}' directory.")
print("You can now upload this file to the CodaBench competition.")