import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# paths
PARTICIPANT_DATA_PATH = './participant_data/'
NEW_DATA_DIR = '0.8_new_participant_data'
os.makedirs(NEW_DATA_DIR, exist_ok=True)

# event pairs
index_events = ["Borrow", "Deposit", "Repay", "Withdraw"]
outcome_events = index_events + ["Liquidated"]
event_pairs = [(i, o) for i in index_events for o in outcome_events if i != o]

# loop through datasets
for index_event, outcome_event in event_pairs:
    print(f"\n{'='*50}")
    print(f"Processing for: {index_event} -> {outcome_event}")
    print(f"{'='*50}")

    dataset_path = os.path.join(index_event, outcome_event)
    
    try:
        train_df = pd.read_csv(os.path.join(PARTICIPANT_DATA_PATH, dataset_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(PARTICIPANT_DATA_PATH, dataset_path, 'test_features.csv'))

        print(f"Original train shape: {train_df.shape}, test shape: {test_df.shape}")

        # separate target and ID columns
        train_target = train_df['status'] if 'status' in train_df.columns else None
        keep_cols = ['id', 'user', 'pool', 'Index Event', 'Outcome Event', 'type', 'timestamp', 'timeDiff', 'status']
        
        train_meta = train_df[[c for c in keep_cols if c in train_df.columns]]
        test_meta = test_df[[c for c in keep_cols if c in test_df.columns]]
        
        # drop metadata columns from feature dataframes
        train_features = train_df.drop(columns=[c for c in keep_cols if c in train_df.columns], errors='ignore')
        test_features = test_df.drop(columns=[c for c in keep_cols if c in test_df.columns], errors='ignore')
        
        # mark which rows are train/test
        train_features['_is_train'] = 1
        test_features['_is_train'] = 0
        
        # combine for consistent preprocessing
        combined = pd.concat([train_features, test_features], ignore_index=True)
        
        # -------------------------------
        # Interaction Features
        # -------------------------------
        interaction_pairs = [
            ('userBorrowSum', 'userBorrowAvgAmountUSD'),
            ('userActiveDaysWeekly', 'userRepayCount'),
            ('userLiquidationCount', 'userRepayAvgAmountUSD')
        ]
        for a, b in interaction_pairs:
            if a in combined.columns and b in combined.columns:
                combined[f'{a}_x_{b}'] = combined[a] * combined[b]

        # -------------------------------
        # Ratio Features
        # -------------------------------
        if 'userRepaySumUSD' in combined.columns and 'userBorrowSumUSD' in combined.columns:
            combined['userRepayEfficiency'] = combined['userRepaySumUSD'] / np.maximum(combined['userBorrowSumUSD'], 1)
        
        if 'marketBorrowSum' in combined.columns and 'marketDepositSum' in combined.columns:
            combined['marketStress'] = combined['marketBorrowSum'] / np.maximum(combined['marketDepositSum'], 1)
        
        if 'userWithdrawSumUSD' in combined.columns and 'userDepositSumUSD' in combined.columns:
            combined['userWithdrawalRatio'] = combined['userWithdrawSumUSD'] / np.maximum(combined['userDepositSumUSD'], 1)

        # -------------------------------
        # Log Features
        # -------------------------------
        log_features = ['amountUSD', 'userBorrowSum', 'userRepaySumUSD', 'marketBorrowSum']
        for f in log_features:
            if f in combined.columns:
                combined[f'log_{f}'] = np.log1p(combined[f])

        # -------------------------------
        # Categorical Encoding
        # -------------------------------
        for cat_col in ['userCoinTypeMode', 'coinType']:
            if cat_col in combined.columns:
                combined = pd.get_dummies(combined, columns=[cat_col], drop_first=True)

        if 'pool' in combined.columns:
            le = LabelEncoder()
            combined['pool_encoded'] = le.fit_transform(combined['pool'].astype(str))
            combined = combined.drop('pool', axis=1)

        # fill missing values before scaling
        combined = combined.fillna(0)

        # -------------------------------
        # PCA on numeric columns
        # -------------------------------
        # get numeric columns (excluding the marker)
        numeric_cols = combined.select_dtypes(include=np.number).columns.tolist()
        if '_is_train' in numeric_cols:
            numeric_cols.remove('_is_train')

        if numeric_cols:
            # split back to train/test
            train_mask = combined['_is_train'] == 1
            train_numeric = combined.loc[train_mask, numeric_cols]
            test_numeric = combined.loc[~train_mask, numeric_cols]
            
            # fit scaler on train only
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_numeric)
            test_scaled = scaler.transform(test_numeric)
            
            # fit PCA on train only
            pca = PCA(n_components=0.95)
            train_pca = pca.fit_transform(train_scaled)
            test_pca = pca.transform(test_scaled)
            
            # create PCA dataframes
            pca_cols = [f'PCA_{i}' for i in range(train_pca.shape[1])]
            train_pca_df = pd.DataFrame(train_pca, columns=pca_cols)
            test_pca_df = pd.DataFrame(test_pca, columns=pca_cols)
            
            # combine with metadata
            train_final = pd.concat([train_meta.reset_index(drop=True), train_pca_df], axis=1)
            test_final = pd.concat([test_meta.reset_index(drop=True), test_pca_df], axis=1)
        else:
            train_final = train_meta
            test_final = test_meta

        print(f"Processed train shape: {train_final.shape}, test shape: {test_final.shape}")

        # save
        save_dir = os.path.join(NEW_DATA_DIR, index_event, outcome_event)
        os.makedirs(save_dir, exist_ok=True)
        train_final.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
        test_final.to_csv(os.path.join(save_dir, 'test_features.csv'), index=False)

        print(f"Saved processed files to {save_dir}")

    except FileNotFoundError:
        print(f"Data not found for {dataset_path}. Skipping.")
        continue
    except Exception as e:
        print(f"Error processing {dataset_path}: {str(e)}")
        continue

print("\nAll data files processed and saved.")