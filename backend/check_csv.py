import pandas as pd

# Read the CSV file
try:
    STOCK_DATA = pd.read_csv('stock_listings.csv')
    
    # Print column names
    print("\nCSV columns:")
    print(STOCK_DATA.columns.tolist())
    
    # Print first 5 rows
    print("\nFirst few rows:")
    print(STOCK_DATA.head())
    
except Exception as e:
    print(f"Error reading CSV: {e}")
