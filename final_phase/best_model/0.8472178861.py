## --- ENSEMBLE XGBOOST WITH MULTIPLE STRATEGIES --- ##
import pandas as pd
import numpy as np
import os
import shutil
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, List
import sys

sys.path.append('.')

from improved_preprocess import preprocess

class EnsembleXGBoost:
    def __init__(self, index_event: str, outcome_event: str):
        self.index_event = index_event
        self.outcome_event = outcome_event
        self.models = []
        self.weights = []
        
    def add_model(self, params: Dict, rounds: int, weight: float = 1.0):
        """Add a model configuration to the ensemble"""
        self.models.append((params, rounds))
        self.weights.append(weight)
    
    def train_ensemble(self, X_train, y_train):
        """Train all models in the ensemble"""
        trained_models = []
        
        for i, ((params, rounds), weight) in enumerate(zip(self.models, self.weights)):
            print(f"  Training model {i+1}/{len(self.models)} (weight: {weight})")
            
            # Prepare target
            y_train_xgb = np.where(
                y_train['status'] == 1,
                np.log(y_train['timeDiff'] + 1),
                -np.log(y_train['timeDiff'] + 1)
            )
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train_xgb)
            
            # Train model
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=rounds,
                evals=[(dtrain, 'train')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            trained_models.append((model, weight))
        
        return trained_models
    
    def predict_ensemble(self, trained_models, X_test):
        """Make ensemble predictions"""
        dtest = xgb.DMatrix(X_test)
        predictions = []
        
        for model, weight in trained_models:
            pred = -model.predict(dtest)
            predictions.append(pred * weight)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred

def create_ensemble_strategies(index_event: str, outcome_event: str) -> EnsembleXGBoost:
    """Create ensemble strategies based on transition performance"""
    
    ensemble = EnsembleXGBoost(index_event, outcome_event)
    
    # Strategy 1: Conservative (high regularization)
    conservative_params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': 4, 'eta': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'min_child_weight': 5, 'lambda': 3.0, 'alpha': 0.5, 'gamma': 0.2, 'seed': 42
    }
    
    # Strategy 2: Aggressive (low regularization)
    aggressive_params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': 8, 'eta': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.9,
        'min_child_weight': 1, 'lambda': 1.0, 'alpha': 0.1, 'gamma': 0.0, 'seed': 43
    }
    
    # Strategy 3: Balanced (medium regularization)
    balanced_params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': 6, 'eta': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 3, 'lambda': 2.0, 'alpha': 0.3, 'gamma': 0.1, 'seed': 44
    }
    
    # Determine weights based on transition performance
    if outcome_event in ['Liquidated', 'Repay', 'Borrow'] and index_event == 'Withdraw':
        # Strong performers - more aggressive
        ensemble.add_model(aggressive_params, 900, 0.5)
        ensemble.add_model(balanced_params, 1000, 0.3)
        ensemble.add_model(conservative_params, 1000, 0.2)
    elif index_event in ['Repay', 'Borrow'] and outcome_event in ['Withdraw', 'Liquidated']:
        # Weak performers - more conservative
        ensemble.add_model(conservative_params, 1500, 0.4)
        ensemble.add_model(balanced_params, 1300, 0.4)
        ensemble.add_model(aggressive_params, 1100, 0.2)
    else:
        # Moderate performers - balanced approach
        ensemble.add_model(balanced_params, 1100, 0.4)
        ensemble.add_model(aggressive_params, 900, 0.3)
        ensemble.add_model(conservative_params, 1200, 0.3)
    
    return ensemble

## --- DEFINE PATH --- ##
TRAIN_DATA_PATH = '/home/dgxuser40/manjil/finsurv/participant_data'
TEST_DATA_PATH = '/home/dgxuser40/manjil/finsurv/test_features'
SUBMISSION_DIR = f'more200_rounds_0.84630_ensemble_xgb'
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
        train_df = pd.read_csv(os.path.join(TRAIN_DATA_PATH, dataset_path, 'train.csv'))
        test_features_df = pd.read_csv(os.path.join(TEST_DATA_PATH, dataset_path, 'test_features.csv'))
        print(f"No of features in train dataset: {len(train_df.keys())}")
        print(f"No of features in holdout test dataset: {len(test_features_df.keys())}")
    except FileNotFoundError as e:
        print(f"Data not found for {dataset_path}. Skipping.")
        continue
        
    # Use transition-specific preprocessing
    X_train, y_train, X_test_processed = preprocess(
        train_df, test_features_df, index_event, outcome_event
    )

    try:
        # Create ensemble for this transition
        ensemble = create_ensemble_strategies(index_event, outcome_event)
        
        print(f"Training ensemble with {len(ensemble.models)} models...")
        trained_models = ensemble.train_ensemble(X_train, y_train)
        
        print("Generating ensemble predictions...")
        predictions = ensemble.predict_ensemble(trained_models, X_test_processed)
        
        # save predictions to a CSV file
        prediction_filename = dataset_path.replace(os.sep, '_') + '.csv'
        prediction_save_path = os.path.join(SUBMISSION_DIR, prediction_filename)
        pd.DataFrame(predictions).to_csv(prediction_save_path, header=False, index=False)
        print(f"  -> Predictions saved to {prediction_save_path}")
        
    except Exception as e:
        print(f"\nERROR: The ensemble for {dataset_path} failed to train.")
        print(f"Details: {e}")

print("\n\nAll prediction files have been generated.")

## --- CREATE SUBMISSION FOLDER --- ##
output_zip_filename = f'more200_rounds_0.84630_ensemble_xgb'
shutil.make_archive(output_zip_filename, 'zip', SUBMISSION_DIR)
print(f"Successfully created '{output_zip_filename}.zip' from the '{SUBMISSION_DIR}' directory.")
print("You can now upload this file to the CodaBench competition.")
