from flask import request, jsonify
import pandas as pd
import os

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load NASDAQ listings
try:
    nasdaq_df = pd.read_csv(os.path.join(current_dir, 'nasdaq-listed.csv'))
except FileNotFoundError:
    nasdaq_df = pd.DataFrame(columns=['Symbol', 'Name', 'Industry'])
    print(f"Error loading NASDAQ stocks: File not found at {os.path.join(current_dir, 'nasdaq-listed.csv')}")