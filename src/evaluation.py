import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_final_model(model, name, X_test, y_test):
    """
    Calculates and prints final evaluation metrics for a given model.
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Final Metrics for {name} ---")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  R-squared (R²): {r2:.4f}")
    
    return y_pred, {'mae': mae, 'rmse': rmse, 'r2': r2}

def plot_results(y_test, y_pred_mlp):
    """
    Generates and saves Predicted vs. Actual and Residuals plots.
    """
    print("\nGenerating and saving plots...")
    
    # 1. Predicted vs. Actual Plot
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test.values, label='Actual Sales', color='blue', alpha=0.7)
    plt.plot(y_test.index, y_pred_mlp, label='Predicted Sales (MLP)', color='red', linestyle='--')
    plt.title('Actual vs. Predicted Sales (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/predictions_vs_actual.png')
    
    # 2. Residuals Distribution Plot
    residuals = y_test - y_pred_mlp
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title('Distribution of Residuals (Actual - Predicted)')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('figures/residuals_distribution.png')
    
    print("Plots saved to 'figures/' directory.")