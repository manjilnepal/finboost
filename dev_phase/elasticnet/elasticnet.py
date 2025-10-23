## --- IMPORT LIBRARIES --- ##
import pandas as pd
import numpy as np
import os
import shutil
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append('../../utilities')

from preprocess import preprocess

ALPHA = 0.1
L1_RATIO = 0.5


## --- DEFINE PATH --- ##
PARTICIPANT_DATA_PATH = '../../participant_data'
SUBMISSION_DIR = f'{ALPHA}_{L1_RATIO}_elasticnet'
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
        print("Training model...")
        
        # Train Elastic Net
        model = ElasticNet(alpha=ALPHA, l1_ratio=L1_RATIO, random_state=42)
        model.fit(X_train, y_train['timeDiff'].values)

        print("Model trained successfully.")

        # --- generate and save predictions ---
        print(f"Generating predictions for {dataset_path}...")
        predictions = -model.predict(X_test_processed)
        
        # save predictions to a CSV file
        prediction_filename = dataset_path.replace(os.sep, '_') + '.csv'
        prediction_save_path = os.path.join(SUBMISSION_DIR, prediction_filename)
        pd.DataFrame(predictions).to_csv(prediction_save_path, header=False, index=False)
        print(f"  -> Predictions saved to {prediction_save_path}")
        print(f"  -> Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        
    except Exception as e:
        print(f"\nERROR: The model for {dataset_path} failed to train.")
        print(f"Details: {e}")

print("\n\nAll prediction files have been generated.")


## --- CREATE SUBMISSION FOLDER --- ##
output_zip_filename = f'{ALPHA}_{L1_RATIO}_elasticnet'
shutil.make_archive(output_zip_filename, 'zip', SUBMISSION_DIR)
print(f"Successfully created '{output_zip_filename}.zip' from the '{SUBMISSION_DIR}' directory.")
print("You can now upload this file to the CodaBench competition.")