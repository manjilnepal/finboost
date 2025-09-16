import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


def smart_one_hot_encode(
    df: pd.DataFrame, top_categories: dict = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Performs one-hot encoding on categorical features, grouping rare categories into 'Other'.
    """
    df_copy = df.copy()

    if top_categories is None:
        top_categories = {}

    categorical_cols = df_copy.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        if col not in top_categories:
            top_cats = df_copy[col].value_counts().nlargest(10).index.tolist()
            top_categories[col] = top_cats

        all_categories = top_categories[col] + ["Other"]
        df_copy[col] = pd.Categorical(df_copy[col], categories=all_categories).fillna(
            "Other"
        )

    df_encoded = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=False)

    return df_encoded, top_categories


def preprocess(
    train_df_with_labels: pd.DataFrame, test_features_df: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses data for the competition.
    """
    # --- 1. Drop high-cardinality identifiers ---
    cols_to_drop = [
        "user",
        "pool",
        "Index Event",
        "Outcome Event",
        "type",
        "timestamp",
    ]
    train_features = train_df_with_labels.drop(columns=cols_to_drop, errors="ignore")
    train_labels = train_df_with_labels[["timeDiff", "status"]]
    train_features = train_features.drop(
        columns=["timeDiff", "status"], errors="ignore"
    )

    if test_features_df is not None:
        test_features_df = test_features_df.drop(columns=cols_to_drop, errors="ignore")

    # --- 2. Smart One-Hot Encode ---
    train_features_encoded, top_categories = smart_one_hot_encode(train_features)

    # --- 3. Scale numerical features ---
    scaler = StandardScaler()
    numerical_cols = train_features_encoded.select_dtypes(include=np.number).columns
    train_features_encoded[numerical_cols] = scaler.fit_transform(
        train_features_encoded[numerical_cols]
    )

    # --- 4. Remove Zero-Variance Features ---
    # This is the new step to prevent matrix singularity
    variances = train_features_encoded.var()
    zero_variance_cols = variances[variances == 0].index
    if not zero_variance_cols.empty:
        print(
            f"  - Dropping {len(zero_variance_cols)} zero-variance columns: {zero_variance_cols.tolist()}"
        )
        train_features_encoded = train_features_encoded.drop(columns=zero_variance_cols)

    X_train_processed = train_features_encoded.fillna(0)
    X_test_processed = None

    # --- Process Test Set if Provided ---
    if test_features_df is not None:
        test_features_encoded, _ = smart_one_hot_encode(
            test_features_df, top_categories=top_categories
        )
        test_features_encoded[numerical_cols] = scaler.transform(
            test_features_encoded[numerical_cols]
        )

        # Align columns after all transformations
        final_cols = X_train_processed.columns
        test_features_final = pd.DataFrame(
            columns=final_cols, index=test_features_encoded.index
        )

        for col in final_cols:
            if col in test_features_encoded:
                test_features_final[col] = test_features_encoded[col]

        X_test_processed = test_features_final.fillna(0)

    return X_train_processed, train_labels, X_test_processed
