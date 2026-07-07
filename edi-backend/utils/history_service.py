# edi-backend/utils/history_service.py (FIXED)

import pandas as pd
import os
import ast

HISTORY_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'history.csv')

def get_all_history():
    """
    Reads all records from history CSV and returns as list of dicts
    Properly parses SHAP scores from string back to list
    """
    if not os.path.exists(HISTORY_PATH):
        return []
    
    try:
        df = pd.read_csv(HISTORY_PATH)
        
        # Convert SHAP scores from string to list
        if 'shap_scores' in df.columns:
            df['shap_scores'] = df['shap_scores'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        
        # Format dates for display
        if 'date' in df.columns:
            df['date_local'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df.to_dict(orient="records")
    
    except Exception as e:
        print(f"❌ Error loading history: {e}")
        return []