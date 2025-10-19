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
        self.use_rank_blend = True  # rank-average improves c-index robustness
        
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
                np.log(y_train['timeDiff'] + 1.0),
                -np.log(y_train['timeDiff'] + 1.0)
            )
            
            # Create DMatrix
            # simple, consistent split for early stopping
            n = X_train.shape[0]
            idx = np.arange(n)
            rs = np.random.RandomState(2025)
            rs.shuffle(idx)
            cut = int(0.9 * n)
            tr_idx, va_idx = idx[:cut], idx[cut:]

            dtrain = xgb.DMatrix(X_train.iloc[tr_idx], label=y_train_xgb[tr_idx])
            dvalid = xgb.DMatrix(X_train.iloc[va_idx], label=y_train_xgb[va_idx])
            
            # Train model
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=rounds,
                evals=[(dtrain, 'train'), (dvalid, 'valid')],
                early_stopping_rounds=100,
                verbose_eval=50
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
        
        if self.use_rank_blend:
            # Convert each prediction to rank to align with c-index optimization
            ranked = []
            for pred in predictions:
                # rankdata without scipy: argsort twice
                order = np.argsort(pred)
                ranks = np.empty_like(order)
                ranks[order] = np.arange(len(pred))
                ranked.append(ranks.astype(np.float32))
            ensemble_pred = np.average(ranked, axis=0, weights=self.weights)
        else:
            # Weighted average of raw scores
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
        'colsample_bylevel': 0.8, 'colsample_bynode': 0.8,
        'min_child_weight': 6, 'lambda': 3.5, 'alpha': 0.6, 'gamma': 0.2, 'seed': 42
    }
    
    # Strategy 2: Aggressive (low regularization)
    aggressive_params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': 8, 'eta': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.9,
        'colsample_bylevel': 0.9, 'colsample_bynode': 0.9,
        'min_child_weight': 1, 'lambda': 1.0, 'alpha': 0.1, 'gamma': 0.0, 'seed': 43
    }
    
    # Strategy 3: Balanced (medium regularization)
    balanced_params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': 6, 'eta': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'colsample_bylevel': 0.85, 'colsample_bynode': 0.85,
        'min_child_weight': 3, 'lambda': 2.2, 'alpha': 0.35, 'gamma': 0.1, 'seed': 44
    }
    
    # Determine weights based on transition performance
    if outcome_event in ['Liquidated', 'Repay', 'Borrow'] and index_event == 'Withdraw':
        # Strong performers - more aggressive
        ensemble.add_model(aggressive_params, 800, 0.5)
        ensemble.add_model(balanced_params, 900, 0.3)
        ensemble.add_model(conservative_params, 1000, 0.2)
    elif index_event == 'Repay' and outcome_event in ['Withdraw', 'Liquidated']:
        # Weak performers from results - prioritize regularization and more rounds
        tuned = dict(conservative_params)
        tuned.update({'eta': 0.03, 'max_depth': 5, 'lambda': 4.0, 'alpha': 0.8, 'seed': 2025})
        ensemble.add_model(tuned, 1400, 0.45)
        ensemble.add_model(balanced_params, 1200, 0.35)
        ensemble.add_model(aggressive_params, 800, 0.20)
    elif index_event == 'Borrow' and outcome_event in ['Withdraw', 'Liquidated']:
        # Also relatively weak - favor balanced + conservative with more trees
        ensemble.add_model(balanced_params, 1300, 0.45)
        ensemble.add_model(conservative_params, 1400, 0.35)
        ensemble.add_model(aggressive_params, 800, 0.20)
    else:
        # Moderate performers - balanced approach
        ensemble.add_model(balanced_params, 900, 0.4)
        ensemble.add_model(aggressive_params, 700, 0.3)
        ensemble.add_model(conservative_params, 1000, 0.3)
    
    return ensemble

## --- DEFINE PATH --- ##
PARTICIPANT_DATA_PATH = '/home/dgxuser40/manjil/finsurv/participant_data'
SUBMISSION_DIR = f'ensemble_xgb_improved'
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
output_zip_filename = f'ensemble_xgb_improved'
shutil.make_archive(output_zip_filename, 'zip', SUBMISSION_DIR)
print(f"Successfully created '{output_zip_filename}.zip' from the '{SUBMISSION_DIR}' directory.")
print("You can now upload this file to the CodaBench competition.")
