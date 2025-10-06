import pandas as pd

def create_features(df):
    """
    Creates time-series features from the datetime index.
    """
    print("Creating features...")
    df = df.copy()
    
    # Calendar features
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    
    # Lag features (sales from previous periods)
    # Using .shift(1) to ensure we only use past data
    df['sales_lag_1'] = df['Sales'].shift(1) # Previous day's sales
    df['sales_lag_7'] = df['Sales'].shift(7) # Sales from one week ago
    
    # Rolling window features
    # Use .shift(1) to prevent data leakage from the current day's sales
    df['rolling_mean_7'] = df['Sales'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df['Sales'].shift(1).rolling(window=7).std()
    
    # Drop rows with NaN values created by lags/rolling features
    df = df.dropna()
    
    # Separate features (X) from the target (y)
    y = df['Sales']
    X = df.drop(columns=['Sales'])
    
    print("Features created successfully.")
    return X, y