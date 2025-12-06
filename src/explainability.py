"""
Loan Approval Prediction System - Explainability Module
========================================================
SHAP-based model explainability for credit risk decisions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Any, List
import warnings
warnings.filterwarnings('ignore')

def compute_shap_values(model: Any, X: pd.DataFrame, 
                        model_type: str = 'tree') -> tuple:
    """
    Compute SHAP values for the model.
    
    Args:
        model: Trained model
        X: Feature matrix
        model_type: 'tree' for tree-based models, 'kernel' for others
    
    Returns:
        explainer, shap_values
    """
    print("\\n🔍 COMPUTING SHAP VALUES")
    print("=" * 50)
    
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))
    
    shap_values = explainer.shap_values(X)
    
    # For binary classification, take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    print(f"  ✅ SHAP values computed for {X.shape[0]} samples")
    
    return explainer, shap_values

def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame, 
                      save_path: str = None, max_display: int = 15):
    """Plot SHAP summary plot (beeswarm)."""
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title('SHAP Summary Plot - Feature Impact on Loan Approval', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_shap_bar(shap_values: np.ndarray, X: pd.DataFrame, 
                  save_path: str = None, max_display: int = 15):
    """Plot SHAP feature importance bar chart."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type='bar', 
                      max_display=max_display, show=False)
    plt.title('SHAP Feature Importance - Mean |SHAP Value|', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_shap_dependence(shap_values: np.ndarray, X: pd.DataFrame,
                         feature: str, interaction_feature: str = None,
                         save_path: str = None):
    """Plot SHAP dependence plot for a feature."""
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values, X, 
                         interaction_index=interaction_feature, show=False)
    plt.title(f'SHAP Dependence Plot: {feature}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_shap_force_single(explainer, shap_values: np.ndarray, 
                           X: pd.DataFrame, idx: int, 
                           save_path: str = None):
    """Plot SHAP force plot for a single prediction."""
    shap.initjs()
    
    force_plot = shap.force_plot(
        explainer.expected_value if not isinstance(explainer.expected_value, list) 
        else explainer.expected_value[1],
        shap_values[idx],
        X.iloc[idx],
        matplotlib=True,
        show=False
    )
    
    plt.title(f'SHAP Force Plot - Sample {idx}', fontsize=14, fontweight='bold')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_shap_waterfall(explainer, shap_values: np.ndarray, 
                        X: pd.DataFrame, idx: int,
                        save_path: str = None):
    """Plot SHAP waterfall plot for a single prediction."""
    expected_value = (explainer.expected_value if not isinstance(explainer.expected_value, list) 
                      else explainer.expected_value[1])
    
    # Create Explanation object
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=expected_value,
        data=X.iloc[idx].values,
        feature_names=X.columns.tolist()
    )
    
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f'SHAP Waterfall Plot - Sample {idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def get_top_features(shap_values: np.ndarray, X: pd.DataFrame, 
                     n_features: int = 10) -> pd.DataFrame:
    """Get top N most important features based on SHAP values."""
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Mean_SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean_SHAP', ascending=False)
    
    return feature_importance.head(n_features)

def explain_feature_in_business_terms(feature_name: str, shap_impact: float) -> str:
    """Translate SHAP feature impact to business language."""
    business_explanations = {
        'Credit_History': {
            'positive': 'Good credit history significantly increases approval chances',
            'negative': 'Poor/no credit history substantially decreases approval likelihood'
        },
        'Log_Total_Income': {
            'positive': 'Higher total household income supports loan repayment ability',
            'negative': 'Lower income raises concerns about repayment capacity'
        },
        'Loan_to_Income': {
            'positive': 'Low loan-to-income ratio indicates manageable debt burden',
            'negative': 'High loan-to-income ratio suggests over-leveraging risk'
        },
        'CreditHistory_Income': {
            'positive': 'Good credit plus high income is a strong approval signal',
            'negative': 'Poor credit with low income is a high-risk combination'
        },
        'Log_LoanAmount': {
            'positive': 'Moderate loan amounts are within acceptable risk thresholds',
            'negative': 'Very high loan amounts increase default probability'
        },
        'Married_Encoded': {
            'positive': 'Married status suggests dual income stability',
            'negative': 'Single applicants have less income buffer'
        },
        'Education_Encoded': {
            'positive': 'Graduate education indicates stable career prospects',
            'negative': 'Non-graduate status may limit earning potential'
        }
    }
    
    direction = 'positive' if shap_impact > 0 else 'negative'
    
    if feature_name in business_explanations:
        return business_explanations[feature_name][direction]
    else:
        impact_word = 'increases' if shap_impact > 0 else 'decreases'
        return f'{feature_name} {impact_word} approval probability'

def generate_shap_report(shap_values: np.ndarray, X: pd.DataFrame) -> str:
    """Generate a comprehensive SHAP analysis report."""
    top_features = get_top_features(shap_values, X, n_features=10)
    
    report = []
    report.append("\\n📊 SHAP EXPLAINABILITY REPORT")
    report.append("=" * 60)
    report.append("\\n🔝 Top 10 Most Important Features:")
    report.append("-" * 40)
    
    for idx, row in top_features.iterrows():
        feature = row['Feature']
        importance = row['Mean_SHAP']
        avg_impact = shap_values[:, X.columns.get_loc(feature)].mean()
        explanation = explain_feature_in_business_terms(feature, avg_impact)
        
        report.append(f"\\n  {idx+1}. {feature}")
        report.append(f"     Mean |SHAP|: {importance:.4f}")
        report.append(f"     💡 {explanation}")
    
    return "\\n".join(report)
