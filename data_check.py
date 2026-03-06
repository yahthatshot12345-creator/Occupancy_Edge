import pandas as pd
import numpy as np
import os

# Define the path to your data
data_path = 'data/phase1_room_occupancy.csv'

if os.path.exists(data_path):
    print("📂 File found! Loading data...")
    df = pd.read_csv(data_path)
    
    # 1. Check the 'Shape' (Rows vs Columns)
    print(f"✅ Data loaded: {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # 2. Check for missing values (The 'NaN' check)
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️ Warning: Found {missing} missing values. We will need to clean these.")
    else:
        print("💎 Clean data: No missing values found.")
        
    # 3. Show the first few rows to verify headers
    print("\n--- Data Preview ---")
    print(df.head())
    
else:
    print(f"❌ Error: Could not find '{data_path}'. Check your folder structure!")