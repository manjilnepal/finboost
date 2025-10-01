# --- import libraries ---
import pandas as pd
import numpy as np
import os
import shutil
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from sklearn.preprocessing import StandardScaler
from lifelines import WeibullAFTFitter
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging

# Configure basic logging to console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Get a logger instance
logger = logging.getLogger('')

sys.path.append('../utilities')
from preprocess import preprocess

PARTICIPANT_DATA_PATH = '../participant_data/'


index_events = ["Borrow", "Deposit", "Repay", "Withdraw"]
outcome_events = index_events + ["Liquidated"]
event_pairs = []
for index_event in index_events:
    for outcome_event in outcome_events:
        if index_event == outcome_event:
            continue
        event_pairs.append((index_event, outcome_event))

for index_event, outcome_event in event_pairs:
    logger.info(f"{'='*50}")
    logger.info(f"Processing and Predicting for: {index_event} -> {outcome_event}")
    logger.info(f"{'='*50}")
    
    dataset_path = os.path.join(index_event, outcome_event)
    
    # --- Load and Preprocess ---
    try:
        train_df = pd.read_csv(os.path.join(PARTICIPANT_DATA_PATH, dataset_path, 'train.csv'))
        test_features_df = pd.read_csv(os.path.join(PARTICIPANT_DATA_PATH, dataset_path, 'test_features.csv'))
    except FileNotFoundError as e:
        logger.info(f"Data not found for {dataset_path}. Skipping.")
        continue
        
    X_train, y_train, X_test_processed = preprocess(train_df, test_features_df)

    # --- Initally Train Model ---
    try:
        lifelines_train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
        lifelines_train_df = lifelines_train_df.loc[lifelines_train_df['timeDiff'] > 0].copy()

        logger.info("Training model on initial training data...")
        model1 = WeibullAFTFitter(penalizer=0.1)
        model1.fit(lifelines_train_df, duration_col='timeDiff', event_col='status')
        
        summary_df = model1.summary

        hr = summary_df['exp(coef)'].to_list() # hazard ratio
        p_values = summary_df['p'].to_list() # p values

        logger.info("Selecting relevant features...")
        cols_to_keep = []
        for index, row in summary_df.iterrows():
            if row['p'] <= 0.00001: # means only features that change hazard by less than ±20% are dropped
                cols_to_keep.append(index)
        
        col_names = [name for _, name in cols_to_keep]
        cols_to_keep.pop()
        cols_to_keep.pop()
        col_names = [name for _, name in cols_to_keep]

        new_train_df = lifelines_train_df[col_names + ['timeDiff', 'status']]
        new_X_test_processed = X_test_processed[col_names]

        col_selected = new_train_df.keys().to_list()

        logger.info(f"{len(col_selected)} columns selected for {index_event}/{outcome_event}")
        logger.info(f"{col_selected}")

        # ## --- save the new dataset --- ##
        # DATSET_SAVE_DIR = os.path.join('./new_participant_data', index_event, outcome_event)
        # os.makedirs(DATSET_SAVE_DIR, exist_ok=True)
        # train_save_path = os.path.join(DATSET_SAVE_DIR, 'new_train.csv')
        # test_save_path = os.path.join(DATSET_SAVE_DIR, 'new_test.csv')
        # new_train_df.to_csv(train_save_path)
        # new_X_test_processed.to_csv(test_save_path)
        # logger.info(f"Sucessfully saved the new dataset!!!")
        logger.info(f"{'='*50}\n\n")
        
    except (ConvergenceError, ValueError) as e:
        logger.info(f"\nERROR: The model for {dataset_path} failed to train. No prediction file will be created.")
        logger.info(f"Details: {e}")

logger.info("\n\nAll new dataset have been generated.")