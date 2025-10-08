##-- MAIN TO-DO -> DATA PRE-PROCESSING --##
import pandas as pd
import numpy as np
import os
import shutil
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

def preprocess(train_df_with_labels: pd.DataFrame,
               test_features_df: Optional[pd.DataFrame] = None,
               ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
        Preprocesses data for the competition.
    """

    # separate target variables (time, event status) and features
    train_targets = train_df_with_labels[["timeDiff", "status"]]
    train_features = train_df_with_labels.drop(columns=["timeDiff", "status"])

    ### ------------ TRAIN FEATURE ENGINEERING --------- ###
    # Borrow-to-Deposit ratio
    train_features['borrow_to_deposit'] = np.where(train_features['userDepositSum'] > 0,
                                    train_features['userBorrowSum'] / train_features['userDepositSum'], 0)

    # Liquidation-to-Borrow ratio
    train_features['liquidation_to_borrow'] = np.where(train_features['userBorrowSum'] > 0,
                                        train_features['userLiquidationSum'] / train_features['userBorrowSum'], 0)

    # Repay-to-Borrow ratio
    train_features['repay_to_borrow'] = np.where(train_features['userBorrowSum'] > 0,
                                    train_features['userRepaySum'] / train_features['userBorrowSum'], 0)
    
    # clean up infinities and NaNs
    train_features.replace([np.inf, -np.inf], 0, inplace=True)
    train_features.fillna(0, inplace=True)
    print(f"-- Train Features Sucessfully Engineered! --")

    ### --- XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX --- ###


    # columns to drop
    cols_to_drop = ["id", "user", "pool", "Index Event", "Outcome Event", "type", "timestamp"]

    # remove unwanted columns from training features
    train_features = train_features.drop(columns=cols_to_drop, errors="ignore")

    # identify categorical columns
    categorical_cols = train_features.select_dtypes(include=["object", "category"]).columns

    # group rare categories into "Other" (keep only top 10 categories per feature)
    for col in categorical_cols:
        top_categories = train_features[col].value_counts().nlargest(10).index
        train_features[col] = train_features[col].where(train_features[col].isin(top_categories), "Other")

    # one-hot encode categorical variables
    train_features_encoded = pd.get_dummies(train_features, columns=categorical_cols, dummy_na=True, drop_first=True)

    # select numerical columns
    numerical_cols = train_features_encoded.select_dtypes(include=np.number).columns
    
    # standardize numerical features (mean=0, std=1)
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_encoded[numerical_cols])

    # create final processed training features
    train_features_final = pd.DataFrame(train_features_scaled, index=train_features_encoded.index, columns=numerical_cols).fillna(0)

    # drop constant features (zero variance)
    cols_to_keep = train_features_final.columns[train_features_final.var() != 0]
    train_features_final = train_features_final[cols_to_keep]


    test_processed_features = None

    if test_features_df is not None:
        # apply same preprocessing steps to test data
        test_features = test_features_df.drop(columns=cols_to_drop, errors="ignore")

        ### ------------ TRAIN FEATURE ENGINEERING --------- ###
        # Borrow-to-Deposit ratio
        test_features['borrow_to_deposit'] = np.where(test_features['userDepositSum'] > 0,
                                        test_features['userBorrowSum'] / test_features['userDepositSum'], 0)

        # Liquidation-to-Borrow ratio
        test_features['liquidation_to_borrow'] = np.where(test_features['userBorrowSum'] > 0,
                                            test_features['userLiquidationSum'] / test_features['userBorrowSum'], 0)

        # Repay-to-Borrow ratio
        test_features['repay_to_borrow'] = np.where(test_features['userBorrowSum'] > 0,
                                        test_features['userRepaySum'] / test_features['userBorrowSum'], 0)
        
        # clean up infinities and NaNs
        test_features.replace([np.inf, -np.inf], 0, inplace=True)
        test_features.fillna(0, inplace=True)
        print(f"-- Test Features Sucessfully Engineered! --")

        ### --- XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX --- ###

        # handle categorical features in test (align with train top 10)
        for col in categorical_cols:
            top_categories = train_features[col].value_counts().nlargest(10).index
            test_features[col] = test_features[col].where(test_features[col].isin(top_categories), "Other")

        # one-hot encode test features   
        test_features_encoded = pd.get_dummies(test_features, columns=categorical_cols, dummy_na=True, drop_first=True)

        # align test columns with training columns
        train_cols = train_features_encoded.columns
        test_features_aligned = test_features_encoded.reindex(columns=train_cols, fill_value=0)

        # # scale test features using training scaler
        test_features_scaled = scaler.transform(test_features_aligned[numerical_cols])

        # create final processed test features
        test_features_final = pd.DataFrame(test_features_scaled, index=test_features_aligned.index, columns=numerical_cols).fillna(0)

        # drop constant features in test
        test_processed_features = test_features_final[cols_to_keep]

    return train_features_final, train_targets, test_processed_features