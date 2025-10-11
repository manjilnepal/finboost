import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional

def preprocess(train_df_with_labels: pd.DataFrame,
               test_features_df: Optional[pd.DataFrame] = None,
               ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Combined preprocessing with feature engineering + scaling + PCA
    """
    
    # Separate target variables (time, event status) and features
    train_targets = train_df_with_labels[["timeDiff", "status"]].copy()
    train_features = train_df_with_labels.drop(columns=["timeDiff", "status"])
    
    # Columns to drop
    cols_to_drop = ["id", "user", "pool", "Index Event", "Outcome Event", "type", "timestamp"]
    
    # Remove unwanted columns
    train_features = train_features.drop(columns=cols_to_drop, errors="ignore")
    
    # Prepare test features if provided
    if test_features_df is not None:
        test_features = test_features_df.drop(columns=cols_to_drop, errors="ignore")
        
        # Mark train/test
        train_features['_is_train'] = 1
        test_features['_is_train'] = 0
        
        # Combine for consistent feature engineering
        combined = pd.concat([train_features, test_features], ignore_index=True)
    else:
        combined = train_features.copy()
    
    # -------------------------------
    # Feature Engineering
    # -------------------------------
    
    # Interaction Features
    interaction_pairs = [
        ('userBorrowSum', 'userBorrowAvgAmountUSD'),
        ('userActiveDaysWeekly', 'userRepayCount'),
        ('userLiquidationCount', 'userRepayAvgAmountUSD')
    ]
    for a, b in interaction_pairs:
        if a in combined.columns and b in combined.columns:
            combined[f'{a}_x_{b}'] = combined[a] * combined[b]
    
    # Ratio Features
    if 'userRepaySumUSD' in combined.columns and 'userBorrowSumUSD' in combined.columns:
        combined['userRepayEfficiency'] = combined['userRepaySumUSD'] / np.maximum(combined['userBorrowSumUSD'], 1)
    
    if 'marketBorrowSum' in combined.columns and 'marketDepositSum' in combined.columns:
        combined['marketStress'] = combined['marketBorrowSum'] / np.maximum(combined['marketDepositSum'], 1)
    
    if 'userWithdrawSumUSD' in combined.columns and 'userDepositSumUSD' in combined.columns:
        combined['userWithdrawalRatio'] = combined['userWithdrawSumUSD'] / np.maximum(combined['userDepositSumUSD'], 1)
    
    # Log Features
    log_features = ['amountUSD', 'userBorrowSum', 'userRepaySumUSD', 'marketBorrowSum']
    for f in log_features:
        if f in combined.columns:
            combined[f'log_{f}'] = np.log1p(combined[f])
    
    # -------------------------------
    # Categorical Encoding
    # -------------------------------
    categorical_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
    if '_is_train' in categorical_cols:
        categorical_cols.remove('_is_train')
    
    # Group rare categories into "Other" (top 10 only)
    for col in categorical_cols:
        if col in combined.columns:
            top_categories = combined[col].value_counts().nlargest(10).index
            combined[col] = combined[col].where(combined[col].isin(top_categories), "Other")
    
    # One-hot encode
    combined = pd.get_dummies(combined, columns=categorical_cols, dummy_na=True, drop_first=True)
    
    # Fill missing values
    combined = combined.fillna(0)
    
    # -------------------------------
    # Split and Scale
    # -------------------------------
    
    if test_features_df is not None:
        # Split back to train/test
        train_mask = combined['_is_train'] == 1
        train_data = combined[train_mask].drop('_is_train', axis=1)
        test_data = combined[~train_mask].drop('_is_train', axis=1)
    else:
        train_data = combined
        test_data = None
    
    # Get numeric columns
    numerical_cols = train_data.select_dtypes(include=np.number).columns.tolist()
    
    if numerical_cols:
        # Fit scaler on train
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data[numerical_cols])
        
        # Remove constant features (zero variance)
        variances = np.var(train_scaled, axis=0)
        cols_to_keep_idx = variances != 0
        train_scaled = train_scaled[:, cols_to_keep_idx]
        kept_cols = [numerical_cols[i] for i, keep in enumerate(cols_to_keep_idx) if keep]
        
        # Fit PCA on train
        pca = PCA(n_components=0.95)  # Keep 95% variance
        train_pca = pca.fit_transform(train_scaled)
        
        # Create final train dataframe
        pca_cols = [f'PCA_{i}' for i in range(train_pca.shape[1])]
        train_features_final = pd.DataFrame(train_pca, index=train_data.index, columns=pca_cols).fillna(0)
        
        # Process test if provided
        test_processed_features = None
        if test_data is not None:
            test_scaled = scaler.transform(test_data[numerical_cols])
            test_scaled = test_scaled[:, cols_to_keep_idx]
            test_pca = pca.transform(test_scaled)
            test_processed_features = pd.DataFrame(test_pca, index=test_data.index, columns=pca_cols).fillna(0)
    else:
        train_features_final = pd.DataFrame(index=train_data.index)
        test_processed_features = pd.DataFrame(index=test_data.index) if test_data is not None else None
    
    return train_features_final, train_targets, test_processed_features