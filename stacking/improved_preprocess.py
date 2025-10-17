##-- IMPROVED PREPROCESSING WITH TRANSITION-SPECIFIC FEATURES --##
import pandas as pd
import numpy as np
import os
import shutil
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Optional


def add_transition_specific_features(df, index_event, outcome_event):
    """Add transition-specific features for better C-index"""
    
    # Base survival features
    df = add_survival_features(df)
    
    # Risk scoring features
    df['user_risk_score'] = (
        df['userLiquidationCount'] * 0.3 +
        (df['userBorrowSumUSD'] / (df['userDepositSumUSD'] + 1)) * 0.4 +
        (df['userBorrowCount'] / (df['userSecondsSinceFirstTransaction'] + 1)) * 0.3
    )
    
    # Transition-specific risk indicators
    if outcome_event == 'Liquidated':
        # Liquidation-specific features
        df['liquidation_risk_score'] = (
            df['userLiquidationCount'] * 0.4 +
            (df['userBorrowSumUSD'] / (df['userDepositSumUSD'] + 1)) * 0.3 +
            df['user_risk_score'] * 0.3
        )
        df['collateral_health'] = df['userDepositSumUSD'] / (df['userBorrowSumUSD'] + 1)
        df['liquidation_proximity'] = 1 / (df['collateral_health'] + 0.1)
        df['debt_to_collateral'] = df['userBorrowSumUSD'] / (df['userDepositSumUSD'] + 0.1)
        df['liquidation_history'] = df['userLiquidationCount'] / (df['userBorrowCount'] + 1)
        
        # Market pressure
        df['market_liquidity_pressure'] = (
            df['marketLiquidationCount'] / (df['marketBorrowCount'] + 1)
        )
        
    elif outcome_event == 'Repay':
        # Repay-specific features
        df['repay_urgency'] = df['userBorrowSumUSD'] / (df['userDepositSumUSD'] + 1)
        df['debt_maturity'] = df['userSecondsSinceFirstTransaction'] / (df['userBorrowCount'] + 1)
        df['repay_capacity'] = df['userDepositSumUSD'] / (df['userBorrowSumUSD'] + 1)
        df['repay_habit'] = df['userRepayCount'] / (df['userBorrowCount'] + 1)
        df['repay_speed'] = df['userRepayCount'] / (df['userSecondsSinceFirstTransaction'] + 1)
        df['outstanding_debt'] = df['userBorrowSumUSD'] - df['userRepaySumUSD']
        df['repayment_ratio'] = df['userRepaySumUSD'] / (df['userBorrowSumUSD'] + 1)
        
    elif outcome_event == 'Withdraw':
        # Withdraw-specific features
        df['withdraw_opportunity'] = df['userDepositSumUSD'] / (df['marketDepositSumUSD'] + 1)
        df['withdraw_timing'] = df['timeOfDay'] * df['dayOfWeek'] / 7
        df['liquidity_preference'] = df['userWithdrawCount'] / (df['userDepositCount'] + 1)
        df['withdrawal_habit'] = df['userWithdrawCount'] / (df['userSecondsSinceFirstTransaction'] + 1)
        df['deposit_withdrawal_ratio'] = df['userWithdrawSumUSD'] / (df['userDepositSumUSD'] + 1)
        df['available_balance'] = df['userDepositSumUSD'] - df['userWithdrawSumUSD']
        df['market_withdraw_rate'] = df['marketWithdrawCount'] / (df['marketDepositCount'] + 1)
        
    elif outcome_event == 'Deposit':
        # Deposit-specific features
        df['deposit_attractiveness'] = df['marketDepositAvgAmountUSD'] / (df['userDepositAvgAmountUSD'] + 1)
        df['deposit_timing'] = df['isWeekend'] * df['is_business_hours']
        df['deposit_momentum'] = df['userDepositCount'] / (df['userSecondsSinceFirstTransaction'] + 1)
        df['deposit_frequency'] = df['userActiveDaysWeekly'] * df['userDepositCount']
        df['market_deposit_share'] = df['userDepositSumUSD'] / (df['marketDepositSumUSD'] + 1)
        
    elif outcome_event == 'Borrow':
        # Borrow-specific features
        df['borrow_necessity'] = df['userBorrowSumUSD'] / (df['userDepositSumUSD'] + 1)
        df['borrow_capacity'] = df['userDepositSumUSD'] / (df['marketBorrowAvgAmountUSD'] + 1)
        df['borrow_risk_tolerance'] = df['userLiquidationCount'] / (df['userBorrowCount'] + 1)
        df['borrow_frequency'] = df['userBorrowCount'] / (df['userSecondsSinceFirstTransaction'] + 1)
        df['market_borrow_share'] = df['userBorrowSumUSD'] / (df['marketBorrowSumUSD'] + 1)
    
    # Advanced user behavior features
    df['user_activity_diversity'] = (
        df['userBorrowCount'] + df['userDepositCount'] + 
        df['userRepayCount'] + df['userWithdrawCount']
    ) / (df['userSecondsSinceFirstTransaction'] + 1)
    
    df['user_transaction_balance'] = (
        (df['userDepositCount'] + df['userRepayCount']) - 
        (df['userBorrowCount'] + df['userWithdrawCount'])
    )
    
    # Market engagement
    df['market_engagement'] = (
        df['userBorrowAvgAmount'] / (df['marketBorrowAvgAmount'] + 1) +
        df['userDepositAvgAmount'] / (df['marketDepositAvgAmount'] + 1)
    ) / 2
    
    # Temporal risk features
    df['weekend_risk'] = df['isWeekend'] * df['userLiquidationCount']
    df['month_end_risk'] = df['is_month_end'] * (df['userBorrowSumUSD'] / (df['userDepositSumUSD'] + 1))
    df['weekend_activity'] = df['isWeekend'] * df['user_activity_diversity']
    
    # Volatility features
    df['amount_volatility'] = np.abs(df['amount'] - df['userBorrowAvgAmount']) / (df['userBorrowAvgAmount'] + 1)
    df['market_deviation'] = np.abs(df['amountUSD'] - df['marketBorrowAvgAmountUSD']) / (df['marketBorrowAvgAmountUSD'] + 1)
    
    # User experience level
    df['user_experience'] = np.log1p(df['userSecondsSinceFirstTransaction'])
    df['user_maturity'] = df['userActiveDaysYearly'] / 365.0
    
    # Transaction size features
    df['transaction_size_ratio'] = df['amountUSD'] / (df['userBorrowAvgAmountUSD'] + df['userDepositAvgAmountUSD'] + 1)
    df['is_large_transaction'] = (df['amountUSD'] > df['userBorrowAvgAmountUSD'] * 2).astype(float)
    df['is_small_transaction'] = (df['amountUSD'] < df['userBorrowAvgAmountUSD'] * 0.5).astype(float)
    
    # Market conditions
    df['market_size'] = df['marketBorrowSumUSD'] + df['marketDepositSumUSD']
    df['market_liquidity'] = df['marketDepositSumUSD'] / (df['marketBorrowSumUSD'] + 1)
    
    return df


def add_survival_features(df):
    """Add survival-specific features for better C-index"""
    
    # User risk scoring
    df['liquidation_risk'] = df['userLiquidationCount'] / (df['userBorrowCount'] + df['userDepositCount'] + 1)
    df['repayment_ratio'] = df['userRepaySum'] / (df['userBorrowSum'] + 1)
    df['leverage_ratio'] = df['userBorrowSumUSD'] / (df['userDepositSumUSD'] + 1)
    
    # User activity patterns
    df['activity_volatility'] = df[['userActiveDaysWeekly', 'userActiveDaysMonthly', 'userActiveDaysYearly']].std(axis=1)
    df['transaction_frequency'] = (df['userBorrowCount'] + df['userDepositCount'] + df['userRepayCount'] + df['userWithdrawCount']) / (df['userSecondsSinceFirstTransaction'] + 1)
    
    # Market interaction features
    df['user_market_borrow_ratio'] = df['userBorrowAvgAmount'] / (df['marketBorrowAvgAmount'] + 1)
    df['user_market_deposit_ratio'] = df['userDepositAvgAmount'] / (df['marketDepositAvgAmount'] + 1)
    df['market_volatility_impact'] = np.abs(df['userBorrowAvgAmount'] - df['marketBorrowAvgAmount']) / (df['marketBorrowAvgAmount'] + 1)
    
    # Temporal features
    df['is_business_hours'] = ((df['timeOfDay'] >= 9) & (df['timeOfDay'] <= 17)).astype(float)
    df['is_month_end'] = (df['dayOfMonth'] >= 28).astype(float)
    df['temporal_interaction'] = df['sinDayOfMonth'] * df['cosDayOfWeek'] * df['sinTimeOfDay']
    
    # Hazard rates
    df['borrow_hazard'] = df['userBorrowCount'] / (df['userSecondsSinceFirstTransaction'] + 1)
    df['deposit_hazard'] = df['userDepositCount'] / (df['userSecondsSinceFirstTransaction'] + 1)
    df['repay_hazard'] = df['userRepayCount'] / (df['userSecondsSinceFirstTransaction'] + 1)
    df['withdraw_hazard'] = df['userWithdrawCount'] / (df['userSecondsSinceFirstTransaction'] + 1)
    
    # Activity consistency
    df['activity_consistency'] = df['userActiveDaysWeekly'] / (df['userActiveDaysMonthly'] + 1)
    
    # Net position
    df['net_position_usd'] = df['userDepositSumUSD'] - df['userBorrowSumUSD']
    df['net_position_ratio'] = df['net_position_usd'] / (df['userDepositSumUSD'] + df['userBorrowSumUSD'] + 1)
    
    return df


def preprocess(train_df_with_labels: pd.DataFrame,
               test_features_df: Optional[pd.DataFrame] = None,
               index_event: str = None,
               outcome_event: str = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
        Preprocesses data for the competition with transition-specific features.
    """

    # Add transition-specific features BEFORE separating targets
    train_df_with_features = add_transition_specific_features(
        train_df_with_labels.copy(), index_event, outcome_event
    )
    train_targets = train_df_with_features[["timeDiff", "status"]]
    train_features = train_df_with_features.drop(columns=["timeDiff", "status"])

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
    
    # Use RobustScaler instead of StandardScaler for better handling of outliers
    scaler = RobustScaler()
    train_features_scaled = scaler.fit_transform(train_features_encoded[numerical_cols])

    # create final processed training features
    train_features_final = pd.DataFrame(train_features_scaled, index=train_features_encoded.index, columns=numerical_cols).fillna(0)

    # drop constant features (zero variance)
    cols_to_keep = train_features_final.columns[train_features_final.var() != 0]
    train_features_final = train_features_final[cols_to_keep]

    test_processed_features = None

    if test_features_df is not None:
        # add transition-specific features to test data BEFORE preprocessing
        test_df_with_features = add_transition_specific_features(
            test_features_df.copy(), index_event, outcome_event
        )

        # apply same preprocessing steps to test data
        test_features = test_df_with_features.drop(columns=cols_to_drop, errors="ignore")

        # handle categorical features in test (align with train top 10)
        for col in categorical_cols:
            top_categories = train_features[col].value_counts().nlargest(10).index
            test_features[col] = test_features[col].where(test_features[col].isin(top_categories), "Other")

        # one-hot encode test features   
        test_features_encoded = pd.get_dummies(test_features, columns=categorical_cols, dummy_na=True, drop_first=True)

        # align test columns with training columns
        train_cols = train_features_encoded.columns
        test_features_aligned = test_features_encoded.reindex(columns=train_cols, fill_value=0)

        # scale test features using training scaler
        test_features_scaled = scaler.transform(test_features_aligned[numerical_cols])

        # create final processed test features
        test_features_final = pd.DataFrame(test_features_scaled, index=test_features_aligned.index, columns=numerical_cols).fillna(0)

        # drop constant features in test
        test_processed_features = test_features_final[cols_to_keep]

    return train_features_final, train_targets, test_processed_features
