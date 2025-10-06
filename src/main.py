import os
import pandas as pd
from sklearn.linear_model import Ridge

# Use relative imports for package structure
from .data_loader import load_and_prepare_data
from .feature_engineering import create_features
from .model_trainer import evaluate_baselines, train_mlp_model
from .evaluation import evaluate_final_model, plot_results

def main_pipeline():
    """
    Runs the full sales forecasting pipeline.
    """
    # --- Setup ---
    DATA_FILEPATH = 'data/raw/Superstore.csv'
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # 1. Load and Prepare Data
    daily_sales_df = load_and_prepare_data(DATA_FILEPATH)

    # 2. Feature Engineering
    X, y = create_features(daily_sales_df)

    # 3. Corrected Time-Aware Split (80% train, 20% test)
    # This is a cleaner approach when using GridSearchCV and early_stopping.
    train_size = int(len(X) * 0.80)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    print(f"\nData split chronologically into {len(X_train)} training and {len(X_test)} testing samples.")
    
    # 4. Evaluate Baselines on the new split
    baseline_results = evaluate_baselines(X_train, y_train, X_test, y_test)
    
    # 5. Train and Tune MLP Model using the training set
    mlp_model = train_mlp_model(X_train, y_train)

    # 6. Evaluate Final MLP Model on the hold-out test set
    y_pred_mlp, mlp_metrics = evaluate_final_model(mlp_model, "MLP Regressor", X_test, y_test)
    
    # 7. Generate and Save Plots using the test set results
    plot_results(y_test, y_pred_mlp)
    
    print("\nSales Forecasting Pipeline finished successfully! 📈")

if __name__ == "__main__":
    main_pipeline()