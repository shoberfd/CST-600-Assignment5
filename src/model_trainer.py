from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_baselines(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates Naive and Ridge baseline models.
    """
    print("\n--- Evaluating Baseline Models ---")
    
    # 1. Naive Baseline (predict today's sales are the same as yesterday's)
    y_pred_naive = X_test['sales_lag_1']
    naive_rmse = np.sqrt(mean_squared_error(y_test, y_pred_naive))
    print(f"Naive Baseline (yesterday's sales) RMSE: {naive_rmse:.2f}")
    
    # 2. Ridge Regression Baseline
    # Only use numeric features for the linear model
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    ridge = Ridge()
    ridge.fit(X_train[numeric_features], y_train)
    y_pred_ridge = ridge.predict(X_test[numeric_features])
    ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    print(f"Ridge Regression Baseline RMSE: {ridge_rmse:.2f}")

    return {'naive_rmse': naive_rmse, 'ridge_rmse': ridge_rmse, 'y_pred_ridge': y_pred_ridge}


def train_mlp_model(X_train, y_train):
    """
    Defines and tunes an MLP Regressor using GridSearchCV with TimeSeriesSplit.
    """
    print("\n--- Training and Tuning MLP Regressor ---")
    
    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(random_state=42, max_iter=1500, early_stopping=True))
    ])
    
    # Define a small, sensible parameter grid
    param_grid = {
        'mlp__hidden_layer_sizes': [(50,), (100,), (50, 25)],
        'mlp__alpha': [0.0001, 0.01, 0.1],      # L2 regularization
        'mlp__learning_rate_init': [0.001, 0.01]
    }
    
    # Use TimeSeriesSplit for cross-validation to respect temporal order
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best CV RMSE: {-grid_search.best_score_:.2f}")
    
    return grid_search.best_estimator_