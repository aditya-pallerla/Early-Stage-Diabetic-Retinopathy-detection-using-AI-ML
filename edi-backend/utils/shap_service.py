# edi-backend/utils/shap_service.py (FOR TABULAR MODEL - LightGBM)

import shap
import numpy as np
from typing import List, Dict

# ============================================================================
# SHAP FOR TABULAR MODEL
# ============================================================================
def get_shap_values(
    tab_model,
    clinical_data_array: np.ndarray,
    feature_names: List[str]
) -> List[Dict[str, any]]:
    """
    Compute SHAP values for tabular model (LightGBM)
    
    Args:
        tab_model: Trained LightGBM model
        clinical_data_array: Scaled clinical features [1, 5]
        feature_names: List of feature names
    
    Returns:
        List of dicts with SHAP values
    """
    print("   Computing SHAP values...")
    
    # Create TreeExplainer (fast for tree-based models like LightGBM)
    explainer = shap.TreeExplainer(tab_model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(clinical_data_array)
    
    # For binary classification, shap_values is a list [class0_shap, class1_shap]
    # We want Stage 1 (class 1) contributions
    if isinstance(shap_values, list):
        shap_array = shap_values[1][0]  # Stage 1 SHAP values
    else:
        shap_array = shap_values[0]
    
    # Process results
    results = []
    max_abs_shap = np.max(np.abs(shap_array))
    
    for i, feature_name in enumerate(feature_names):
        raw_shap = float(shap_array[i])
        
        # Scale to 0-100
        if max_abs_shap > 0:
            scaled_score = int(abs(raw_shap) / max_abs_shap * 100)
        else:
            scaled_score = 0
        
        impact = "Increases Risk" if raw_shap > 0 else "Decreases Risk"
        
        results.append({
            "feature": feature_name,
            "shap_value": round(raw_shap, 4),
            "scaled_score": scaled_score,
            "impact": impact,
            "importance_rank": 0
        })
    
    # Sort by importance
    results.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    
    # Add ranks
    for rank, item in enumerate(results, start=1):
        item['importance_rank'] = rank
    
    return results