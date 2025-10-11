## --- IMPORT LIBRARIES --- ##
import pandas as pd
import numpy as np
import os
import shutil
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import sys

sys.path.append('../utilities')
from preprocess import preprocess

## --- OPTIMIZED HYPERPARAMETERS --- ##
BASE_PARAMS = {
    'objective': 'survival:cox',
    'eval_metric': 'cox-nloglik',
    
    # Tree structure - deeper trees for complex financial patterns
    'max_depth': 8,
    
    # Learning rate - balance between speed and accuracy
    'eta': 0.03,
    
    # Sampling parameters - prevent overfitting
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'colsample_bylevel': 0.7,
    
    # Regularization - critical for high-dimensional data
    'min_child_weight': 3,
    'lambda': 2.0,
    'alpha': 0.3,
    'gamma': 0.1,
    
    # Additional parameters for imbalanced data
    'max_delta_step': 1,
    
    'seed': 42,
    'tree_method': 'hist',
    'grow_policy': 'lossguide',
}

NUM_BOOST_ROUND = 800
EARLY_STOPPING_ROUNDS = 50
VALIDATION_SIZE = 0.15

## --- DEFINE PATH --- ##
PARTICIPANT_DATA_PATH = '/home/dgxuser40/manjil/finsurv/participant_data'
SUBMISSION_DIR = f'optimized_xgb_eta{BASE_PARAMS["eta"]}_depth{BASE_PARAMS["max_depth"]}_rounds{NUM_BOOST_ROUND}'
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

print(f"Starting training with optimized hyperparameters:")
print(f"  - max_depth: {BASE_PARAMS['max_depth']}")
print(f"  - eta: {BASE_PARAMS['eta']}")
print(f"  - num_boost_round: {NUM_BOOST_ROUND}")
print(f"  - early_stopping_rounds: {EARLY_STOPPING_ROUNDS}")
print(f"  - validation_size: {VALIDATION_SIZE}")
print(f"\nTotal event pairs to process: {len(event_pairs)}\n")

## --- START TRAINING PROCESS --- ##
results_summary = []

for idx, (index_event, outcome_event) in enumerate(event_pairs, 1):
    print(f"\n{'='*70}")
    print(f"[{idx}/{len(event_pairs)}] Processing: {index_event} -> {outcome_event}")
    print(f"{'='*70}")
    
    dataset_path = os.path.join(index_event, outcome_event)
    
    try:
        train_df = pd.read_csv(os.path.join(PARTICIPANT_DATA_PATH, dataset_path, 'train.csv'))
        test_features_df = pd.read_csv(os.path.join(PARTICIPANT_DATA_PATH, dataset_path, 'test_features.csv'))
    except FileNotFoundError as e:
        print(f"⚠ Data not found for {dataset_path}. Skipping.")
        continue
    
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_features_df)}")
    
    # Preprocess data
    X_train, y_train, X_test_processed = preprocess(train_df, test_features_df)
    
    # Calculate censoring rate for this dataset
    censoring_rate = 1 - (y_train['status'].sum() / len(y_train))
    event_rate = y_train['status'].sum() / len(y_train)
    print(f"  Censoring rate: {censoring_rate:.2%}")
    print(f"  Event rate: {event_rate:.2%}")
    
    try:
        # Create a copy of params for this specific dataset
        params = BASE_PARAMS.copy()
        
        # Adjust scale_pos_weight based on censoring rate
        if censoring_rate > 0:
            params['scale_pos_weight'] = censoring_rate / (1 - censoring_rate + 1e-6)
        else:
            params['scale_pos_weight'] = 1.0
        
        print(f"  scale_pos_weight: {params['scale_pos_weight']:.3f}")
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, 
            test_size=VALIDATION_SIZE, 
            random_state=42,
            stratify=y_train['status']
        )
        
        # Convert to XGBoost format (positive for events, negative for censored)
        y_train_xgb_split = np.where(
            y_train_split['status'] == 1,
            y_train_split['timeDiff'],
            -y_train_split['timeDiff']
        )
        
        y_val_xgb = np.where(
            y_val_split['status'] == 1,
            y_val_split['timeDiff'],
            -y_val_split['timeDiff']
        )
        
        # Convert full training set for final model
        y_train_xgb_full = np.where(
            y_train['status'] == 1,
            y_train['timeDiff'],
            -y_train['timeDiff']
        )
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train_split, label=y_train_xgb_split)
        dval = xgb.DMatrix(X_val_split, label=y_val_xgb)
        dtrain_full = xgb.DMatrix(X_train, label=y_train_xgb_full)
        dtest = xgb.DMatrix(X_test_processed)
        
        print(f"\n  Training model with early stopping...")
        
        # Train model with validation
        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            evals_result=evals_result,
            verbose_eval=False
        )
        
        best_iteration = model.best_iteration
        best_score = model.best_score
        
        print(f"  ✓ Training completed")
        print(f"  Best iteration: {best_iteration}")
        print(f"  Best validation score: {best_score:.6f}")
        
        # Retrain on full training data with optimal number of rounds
        print(f"\n  Retraining on full dataset with {best_iteration} rounds...")
        final_model = xgb.train(
            params,
            dtrain_full,
            num_boost_round=best_iteration,
            verbose_eval=False
        )
        
        # Generate predictions
        print(f"  Generating predictions...")
        predictions = -final_model.predict(dtest)
        
        # Save predictions
        prediction_filename = dataset_path.replace(os.sep, '_') + '.csv'
        prediction_save_path = os.path.join(SUBMISSION_DIR, prediction_filename)
        pd.DataFrame(predictions).to_csv(prediction_save_path, header=False, index=False)
        print(f"  ✓ Predictions saved to {prediction_filename}")
        
        # Store results summary
        results_summary.append({
            'index_event': index_event,
            'outcome_event': outcome_event,
            'train_samples': len(train_df),
            'test_samples': len(test_features_df),
            'censoring_rate': censoring_rate,
            'event_rate': event_rate,
            'best_iteration': best_iteration,
            'best_val_score': best_score,
            'scale_pos_weight': params['scale_pos_weight']
        })
        
    except Exception as e:
        print(f"\n❌ ERROR: The model for {dataset_path} failed to train.")
        print(f"Details: {e}")
        import traceback
        traceback.print_exc()

print("\n\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

# Create summary DataFrame
summary_df = pd.DataFrame(results_summary)
if len(summary_df) > 0:
    print(f"\nSuccessfully trained {len(summary_df)}/{len(event_pairs)} models")
    print(f"\nAverage best iteration: {summary_df['best_iteration'].mean():.0f}")
    print(f"Average validation score: {summary_df['best_val_score'].mean():.6f}")
    print(f"Average censoring rate: {summary_df['censoring_rate'].mean():.2%}")
    
    # Save summary
    summary_path = os.path.join(SUBMISSION_DIR, 'training_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Training summary saved to {summary_path}")
    
    # Print top and bottom performers
    summary_df_sorted = summary_df.sort_values('best_val_score')
    print(f"\nTop 3 best performing models:")
    for i, row in summary_df_sorted.head(3).iterrows():
        print(f"  {row['index_event']:8s} -> {row['outcome_event']:10s}: {row['best_val_score']:.6f}")
    
    print(f"\nTop 3 most challenging models:")
    for i, row in summary_df_sorted.tail(3).iterrows():
        print(f"  {row['index_event']:8s} -> {row['outcome_event']:10s}: {row['best_val_score']:.6f}")

print("\n" + "="*70)
print("CREATING SUBMISSION FILE")
print("="*70)

## --- CREATE SUBMISSION FOLDER --- ##
output_zip_filename = SUBMISSION_DIR
shutil.make_archive(output_zip_filename, 'zip', SUBMISSION_DIR)
print(f"\n✓ Successfully created '{output_zip_filename}.zip'")
print(f"✓ Ready to upload to CodaBench competition")
print("\n" + "="*70)
print("PROCESS COMPLETE")
print("="*70)