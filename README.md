# CST-600-Assignment5

# Neural Network Sales Forecasting for Zenith Electronics

This project develops a short-horizon sales forecasting model using a Multi-Layer Perceptron (MLP) neural network. The model is trained on historical e-commerce data to predict next-day sales for the "Technology" category, aiming to improve inventory management and marketing decisions for Zenith Electronics.

## Business Scenario
As a data scientist at Zenith Electronics, the goal is to produce reliable daily sales predictions. This project prototypes a neural network regressor, from data ingestion and feature engineering to model tuning and evaluation, and provides a clear analysis of its performance against simpler baseline models.

---
## Dataset
* **Source**: [Sample - Superstore Dataset on Kaggle](https://www.kaggle.com/datasets/safavieh/superstore)
* **Preparation**:
  * The dataset was filtered to include sales from the **"Technology"** category only.
  * Sales data was aggregated to a **daily** granularity to create a consistent time series.
* **Forecast Task**: Predict the next day's total sales (1-day horizon, daily granularity).

---
## Environment Setup
Follow these steps to set up the local environment on a Windows machine.

1.  **Clone the Repository**
2.  **Create and Activate the Virtual Environment**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
---
## How to Run
After setting up the environment and placing the `Sample - Superstore.csv` file in the `data/raw/` folder:

1.  Navigate to the project's **root directory** in your terminal.
2.  Run the main script **as a module**:
    ```bash
    python -m src.main
    ```
    > **Note:** Running as a module (`-m`) is required for Python to correctly handle the relative imports between files in the `src` package.

---
## Summary of Decisions & Results
* **Feature Engineering**: To provide the model with historical context, several time-aware features were created:
  * **Calendar Features**: Day of week, month, year, quarter.
  * **Lag Features**: Sales from the previous day (`t-1`) and the same day last week (`t-7`).
  * **Rolling Features**: 7-day rolling mean and standard deviation of past sales.
* **Data Splitting**: A chronological 70/30 train/test split was used to ensure the model was trained only on past data and evaluated on future data, preventing data leakage.
* **Model Tuning**: `GridSearchCV` with `TimeSeriesSplit` was used to find the optimal MLP architecture and regularization. This cross-validation method respects the temporal order of the data.
* **Results**: The tuned **MLP Regressor** significantly outperformed both the Naïve (yesterday's sales) and Ridge Regression baselines on the hold-out test set, demonstrating the value of its non-linear modeling capabilities for this forecasting task.