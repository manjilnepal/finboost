import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_censored

def get_concordance_index(
    test_df: pd.DataFrame, 
    predictions: np.ndarray
) -> float:
    """
    Calculates the concordance index for survival models using scikit-survival.
    """
    event_indicator = test_df['status'].astype(bool)
    event_time = test_df['timeDiff']

    # Handle cases where all events are censored or all are non-censored in the test set
    if len(np.unique(event_indicator)) == 1:
        return 0.5  # Return a neutral score

    c_index, _, _, _, _ = concordance_index_censored(
        event_indicator, event_time, -predictions
    )
    
    return c_index
