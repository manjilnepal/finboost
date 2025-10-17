## --- ENSEMBLE XGBOOST WITH MULTIPLE STRATEGIES --- ##
import pandas as pd
import numpy as np
import os
import shutil
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge, ElasticNet
from typing import Tuple, Optional, Dict, List
from lifelines.utils import concordance_index
import sys

# Ensure local imports work regardless of the working directory
sys.path.append(os.path.dirname(__file__))

from improved_preprocess import preprocess

# -------- Runtime controls for large datasets --------
# Enable FAST_MODE to reduce training time (fewer folds/models/rounds)
FAST_MODE = os.getenv("FINSURV_FAST", "1") == "1"
CPU_THREADS = max(1, (os.cpu_count() or 4) - 0)
N_SPLITS = 5 if FAST_MODE else 7
ROUND_SCALE = 0.65 if FAST_MODE else 1.0
MAX_MODELS_PER_TRANSITION = 4 if FAST_MODE else 6

def scaled_rounds(original_rounds: int) -> int:
    # Ensure a reasonable floor to allow early stopping to activate
    return max(300 if FAST_MODE else 500, int(original_rounds * ROUND_SCALE))

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
        """Train all models on full data (used later for test-time meta)."""
        trained_models = []
        
        # Prepare target with improved transformation
        y_train_xgb = np.where(
            y_train['status'] == 1,
            np.log1p(y_train['timeDiff']),
            -np.log1p(y_train['timeDiff'])
        )
        dtrain_full = xgb.DMatrix(X_train, label=y_train_xgb)
        
        for i, ((params, rounds), weight) in enumerate(zip(self.models, self.weights)):
            print(f"  Training full model {i+1}/{len(self.models)} (weight: {weight})")
            train_params = {**params, 'nthread': CPU_THREADS}
            model = xgb.train(
                train_params,
                dtrain_full,
                num_boost_round=scaled_rounds(rounds),
                evals=[(dtrain_full, 'train')],
                early_stopping_rounds=75,
                verbose_eval=False
            )
            trained_models.append((model, weight))
        return trained_models
    
    def predict_ensemble(self, trained_models, X_test):
        """Make ensemble predictions (weighted average of base models)."""
        dtest = xgb.DMatrix(X_test)
        predictions = []
        for model, weight in trained_models:
            pred = -model.predict(dtest)
            predictions.append(pred * weight)
        return np.average(predictions, axis=0, weights=self.weights)

    def generate_oof_predictions(self, X_train, y_train, n_splits: int = N_SPLITS, random_state: int = 2025):
        """Generate out-of-fold predictions for each base model for meta-learner.
        Returns:
            oof_matrix: shape (n_samples, n_models)
            fold_models: list of lists of trained fold models per base model
            oof_cindex_per_model: list of c-index per base model on OOF
        """
        n_models = len(self.models)
        n_samples = X_train.shape[0]
        oof_matrix = np.zeros((n_samples, n_models), dtype=np.float32)
        fold_models: List[List[xgb.Booster]] = [[] for _ in range(n_models)]

        # Prepare target with improved transformation
        y_train_xgb = np.where(
            y_train['status'] == 1,
            np.log1p(y_train['timeDiff']),
            -np.log1p(y_train['timeDiff'])
        )
        status = y_train['status'].astype(int).values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for m_idx, ((params, rounds), weight) in enumerate(zip(self.models, self.weights)):
            print(f"  OOF for base model {m_idx+1}/{n_models} (weight: {weight})")
            oof_pred_m = np.zeros(n_samples, dtype=np.float32)
            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, status), start=1):
                dtr = xgb.DMatrix(X_train.iloc[tr_idx], label=y_train_xgb[tr_idx])
                dva = xgb.DMatrix(X_train.iloc[va_idx], label=y_train_xgb[va_idx])
            # inject runtime-related params
            train_params = {**params, 'nthread': CPU_THREADS}
            model = xgb.train(
                train_params,
                dtr,
                num_boost_round=scaled_rounds(rounds),
                evals=[(dtr, 'train'), (dva, 'valid')],
                early_stopping_rounds=120,
                verbose_eval=False)
            
            pred_va = -model.predict(dva)
            oof_pred_m[va_idx] = pred_va.astype(np.float32)
            fold_models[m_idx].append(model)
            print(f"    Fold {fold}: valid size={len(va_idx)}")
            oof_matrix[:, m_idx] = oof_pred_m
        
        # Compute per-model OOF c-index for reference
        durations = y_train['timeDiff'].values
        events = y_train['status'].values
        oof_cindex_per_model = []
        for m_idx in range(n_models):
            c = concordance_index(durations, -oof_matrix[:, m_idx], events)
            oof_cindex_per_model.append(c)
        print("Base models OOF c-index:", [f"{c:.4f}" for c in oof_cindex_per_model])

        return oof_matrix, fold_models, oof_cindex_per_model

class RankNormalizer:
    """Fits a percentile mapping on training values and transforms new arrays
    to [0,1] by empirical CDF. Using OOF-fit avoids test leakage and stabilizes
    the meta learner for c-index optimization.
    """
    def __init__(self):
        self._sorted: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "RankNormalizer":
        self._sorted = np.sort(values.astype(np.float64))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self._sorted is None:
            raise RuntimeError("RankNormalizer not fitted")
        ranks = np.searchsorted(self._sorted, values, side="right")
        return (ranks.astype(np.float32) + 0.5) / (self._sorted.size + 1.0)

def rank_normalize(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(arr))
    return (ranks.astype(np.float32) + 0.5) / len(arr)

def to_submission_scores(raw_risk_scores: np.ndarray) -> np.ndarray:
    # Higher risk (shorter time) should have larger submission score
    return -raw_risk_scores

def create_ensemble_strategies(index_event: str, outcome_event: str) -> EnsembleXGBoost:
    """Create improved ensemble strategies with 4 models for better diversity"""
    
    ensemble = EnsembleXGBoost(index_event, outcome_event)
    
    # Base parameters optimized for survival analysis
    base_params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'nthread': CPU_THREADS,
    }
    
    # Strategy 1: Deep trees with strong regularization
    deep_regularized = {
        **base_params,
        'max_depth': 10,
        'eta': 0.015,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'min_child_weight': 5,
        'lambda': 4.0,
        'alpha': 0.8,
        'gamma': 0.3,
        'max_leaves': 63,
        'seed': 42
    }
    
    # Strategy 2: Balanced (medium depth)
    balanced = {
        **base_params,
        'max_depth': 7,
        'eta': 0.025,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'colsample_bylevel': 0.85,
        'colsample_bynode': 0.85,
        'min_child_weight': 3,
        'lambda': 2.5,
        'alpha': 0.4,
        'gamma': 0.15,
        'max_leaves': 127,
        'seed': 43
    }
    
    # Strategy 3: Shallow light (ensemble diversity)
    shallow_light = {
        **base_params,
        'max_depth': 5,
        'eta': 0.04,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'colsample_bylevel': 0.9,
        'colsample_bynode': 0.9,
        'min_child_weight': 2,
        'lambda': 1.5,
        'alpha': 0.2,
        'gamma': 0.05,
        'max_leaves': 31,
        'seed': 44
    }
    
    # Strategy 4: Very deep aggressive
    aggressive = {
        **base_params,
        'max_depth': 12,
        'eta': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'min_child_weight': 1,
        'lambda': 1.0,
        'alpha': 0.1,
        'gamma': 0.0,
        'max_leaves': 255,
        'seed': 45
    }
    
    # Transition-specific configurations
    transition_key = f"{index_event}/{outcome_event}"
    
    # HIGH PERFORMERS
    if transition_key in ['Withdraw/Liquidated', 'Withdraw/Repay', 'Withdraw/Borrow', 'Deposit/Liquidated']:
        configs = [
            (balanced, 1000, 0.25),
            (deep_regularized, 1200, 0.25),
            (aggressive, 900, 0.20),
            (shallow_light, 700, 0.10),
            ({**balanced, 'eta': balanced['eta'] * 0.6}, int(1000 * 1.6), 0.10),
            ({**deep_regularized, 'eta': deep_regularized['eta'] * 0.6}, int(1200 * 1.6), 0.10),
        ]
        for cfg in configs[:MAX_MODELS_PER_TRANSITION]:
            ensemble.add_model(*cfg)
    
    # LOW PERFORMERS - Repay transitions (need most help)
    elif index_event == 'Repay':
        configs = [
            (deep_regularized, 1600, 0.30),
            (balanced, 1400, 0.25),
            (aggressive, 1200, 0.20),
            (shallow_light, 800, 0.10),
            ({**deep_regularized, 'eta': deep_regularized['eta'] * 0.6}, int(1600 * 1.6), 0.10),
            ({**balanced, 'eta': balanced['eta'] * 0.6}, int(1400 * 1.6), 0.05),
        ]
        for cfg in configs[:MAX_MODELS_PER_TRANSITION]:
            ensemble.add_model(*cfg)
    
    # LOW PERFORMERS - Borrow to Withdraw/Liquidated
    elif index_event == 'Borrow' and outcome_event in ['Withdraw', 'Liquidated']:
        configs = [
            (deep_regularized, 1400, 0.30),
            (balanced, 1300, 0.25),
            (aggressive, 1100, 0.20),
            (shallow_light, 700, 0.10),
            ({**deep_regularized, 'eta': deep_regularized['eta'] * 0.6}, int(1400 * 1.6), 0.10),
            ({**balanced, 'eta': balanced['eta'] * 0.6}, int(1300 * 1.6), 0.05),
        ]
        for cfg in configs[:MAX_MODELS_PER_TRANSITION]:
            ensemble.add_model(*cfg)
    
    # MID PERFORMERS
    else:
        configs = [
            (balanced, 1200, 0.25),
            (deep_regularized, 1300, 0.25),
            (aggressive, 1000, 0.20),
            (shallow_light, 700, 0.10),
            ({**balanced, 'eta': balanced['eta'] * 0.6}, int(1200 * 1.6), 0.10),
            ({**deep_regularized, 'eta': deep_regularized['eta'] * 0.6}, int(1300 * 1.6), 0.10),
        ]
        for cfg in configs[:MAX_MODELS_PER_TRANSITION]:
            ensemble.add_model(*cfg)
    
    return ensemble

def meta_params_for_transition(index_event: str, outcome_event: str) -> dict:
    """Heuristic ElasticNet settings per transition.
    Stronger regularization for noisier Repay/* and Borrow->(Withdraw, Liquidated).
    """
    if index_event == 'Repay':
        return dict(alpha=1.5, l1_ratio=0.15)
    if index_event == 'Borrow' and outcome_event in ['Withdraw', 'Liquidated']:
        return dict(alpha=1.2, l1_ratio=0.2)
    if f"{index_event}/{outcome_event}" in ['Withdraw/Liquidated', 'Withdraw/Repay', 'Withdraw/Borrow', 'Deposit/Liquidated']:
        return dict(alpha=0.4, l1_ratio=0.1)
    return dict(alpha=0.8, l1_ratio=0.15)

## --- DEFINE PATH --- ##
PARTICIPANT_DATA_PATH = '/home/dgxuser40/manjil/finsurv/participant_data'
SUBMISSION_DIR = f'stacking_xgb_oof_meta'
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

        # 1) Generate OOF predictions for base models (7-fold for stability)
        print(f"Generating OOF predictions for {len(ensemble.models)} base models...")
        oof_matrix, fold_models, _ = ensemble.generate_oof_predictions(X_train, y_train, n_splits=7, random_state=2025)

        # 2) Fit per-model rank normalizers on OOF and transform OOF to ranks
        normalizers: list[RankNormalizer] = []
        oof_rank_cols = []
        for j in range(oof_matrix.shape[1]):
            rn = RankNormalizer().fit(oof_matrix[:, j])
            normalizers.append(rn)
            oof_rank_cols.append(rn.transform(oof_matrix[:, j]))
        oof_rank = np.column_stack(oof_rank_cols)

        # 3) Train ElasticNet(positive=True) meta-learner on OOF ranks
        y_train_xgb = np.where(
            y_train['status'] == 1,
            np.log1p(y_train['timeDiff']),
            -np.log1p(y_train['timeDiff'])
        )
        meta_cfg = meta_params_for_transition(index_event, outcome_event)
        meta = ElasticNet(alpha=meta_cfg['alpha'], l1_ratio=meta_cfg['l1_ratio'], positive=True, max_iter=2000, random_state=2025)
        meta.fit(oof_rank, y_train_xgb)

        # 4) OOF c-index of meta
        oof_meta = meta.predict(oof_rank)
        c_meta = concordance_index(y_train['timeDiff'].values, -oof_meta, y_train['status'].values)
        print(f"  Meta OOF c-index: {c_meta:.5f}")

        # 5) Train full-data base models (for test-time predictions)
        print(f"Training full-data base models...")
        trained_models = ensemble.train_ensemble(X_train, y_train)

        # 6) Get base predictions on test, transform by OOF-fitted normalizers
        dtest = xgb.DMatrix(X_test_processed)
        test_base = []
        for model, _w in trained_models:
            pred = -model.predict(dtest)
            test_base.append(pred.astype(np.float32))
        test_base = np.column_stack(test_base)
        test_rank_cols = [normalizers[j].transform(test_base[:, j]) for j in range(test_base.shape[1])]
        test_rank = np.column_stack(test_rank_cols)

        # 7) Meta predictions and sign for submission
        test_meta = meta.predict(test_rank)
        predictions = to_submission_scores(test_meta)

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
output_zip_filename = f'stacking_xgb_oof_meta'
shutil.make_archive(output_zip_filename, 'zip', SUBMISSION_DIR)
print(f"Successfully created '{output_zip_filename}.zip' from the '{SUBMISSION_DIR}' directory.")
print("You can now upload this file to the CodaBench competition.")
