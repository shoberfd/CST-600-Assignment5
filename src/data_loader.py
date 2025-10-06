import pandas as pd

def load_and_prepare_data(filepath):
    """
    Loads the Superstore dataset, filters for 'Technology' sales,
    and aggregates them by day.
    """
    print("Loading and preparing data...")
    
    # Load the dataset using default UTF-8 encoding
    df = pd.read_csv(filepath)
    
    # Let pandas automatically infer the datetime format
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    
    # Filter for the 'Technology' category
    df_tech = df[df['Category'] == 'Technology'].copy()
    
    # Set 'Order Date' as the index
    df_tech.set_index('Order Date', inplace=True)
    
    # Resample to get total daily sales, filling missing days with 0
    daily_sales = df_tech['Sales'].resample('D').sum().fillna(0)
    
    print(f"Data prepared: {len(daily_sales)} daily records from {daily_sales.index.min().date()} to {daily_sales.index.max().date()}.")
    return pd.DataFrame(daily_sales)