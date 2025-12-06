"""
Loan Approval Prediction System - Fairness Module
==================================================
Bias detection and fairness analysis for ethical AI.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from sklearn.metrics import confusion_matrix

def compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray, group: np.ndarray) -> Dict:
    """Compute metrics for each group."""
    unique_groups = np.unique(group)
    metrics = {}
    
    for g in unique_groups:
        mask = group == g
        if mask.sum() == 0:
            continue
        
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel() if len(np.unique(y_t)) > 1 else (0, 0, 0, 0)
        
        metrics[g] = {
            'count': mask.sum(),
            'approval_rate': y_p.mean(),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'positive_rate': (tp + fp) / mask.sum()
        }
    
    return metrics

def compute_disparate_impact(group_metrics: Dict, 
                             privileged_group: str, 
                             unprivileged_group: str) -> float:
    """
    Compute Disparate Impact Ratio.
    
    DI = P(Approved | Unprivileged) / P(Approved | Privileged)
    
    A ratio < 0.8 or > 1.25 typically indicates bias.
    """
    if privileged_group not in group_metrics or unprivileged_group not in group_metrics:
        return None
    
    priv_rate = group_metrics[privileged_group]['approval_rate']
    unpriv_rate = group_metrics[unprivileged_group]['approval_rate']
    
    if priv_rate == 0:
        return float('inf')
    
    return unpriv_rate / priv_rate

def compute_equal_opportunity_difference(group_metrics: Dict,
                                         privileged_group: str,
                                         unprivileged_group: str) -> float:
    """
    Compute Equal Opportunity Difference.
    
    EOD = TPR(Unprivileged) - TPR(Privileged)
    
    Values close to 0 indicate fairness.
    """
    if privileged_group not in group_metrics or unprivileged_group not in group_metrics:
        return None
    
    priv_tpr = group_metrics[privileged_group]['true_positive_rate']
    unpriv_tpr = group_metrics[unprivileged_group]['true_positive_rate']
    
    return unpriv_tpr - priv_tpr

def compute_demographic_parity_difference(group_metrics: Dict,
                                          privileged_group: str,
                                          unprivileged_group: str) -> float:
    """
    Compute Statistical/Demographic Parity Difference.
    
    DPD = P(Approved | Unprivileged) - P(Approved | Privileged)
    
    Values close to 0 indicate fairness.
    """
    if privileged_group not in group_metrics or unprivileged_group not in group_metrics:
        return None
    
    priv_rate = group_metrics[privileged_group]['approval_rate']
    unpriv_rate = group_metrics[unprivileged_group]['approval_rate']
    
    return unpriv_rate - priv_rate

def analyze_fairness_for_attribute(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_prob: np.ndarray, protected_attr: np.ndarray,
                                   attr_name: str, 
                                   privileged_group: str,
                                   unprivileged_group: str) -> Dict:
    """Comprehensive fairness analysis for a protected attribute."""
    group_metrics = compute_group_metrics(y_true, y_pred, y_prob, protected_attr)
    
    di = compute_disparate_impact(group_metrics, privileged_group, unprivileged_group)
    eod = compute_equal_opportunity_difference(group_metrics, privileged_group, unprivileged_group)
    dpd = compute_demographic_parity_difference(group_metrics, privileged_group, unprivileged_group)
    
    return {
        'attribute': attr_name,
        'group_metrics': group_metrics,
        'disparate_impact': di,
        'equal_opportunity_diff': eod,
        'demographic_parity_diff': dpd,
        'privileged_group': privileged_group,
        'unprivileged_group': unprivileged_group
    }

def interpret_fairness_metric(metric_name: str, value: float) -> str:
    """Interpret fairness metric value."""
    if metric_name == 'disparate_impact':
        if value is None:
            return "⚠️ Could not compute (insufficient data)"
        elif 0.8 <= value <= 1.25:
            return f"✅ FAIR ({value:.3f}) - Within acceptable range [0.8, 1.25]"
        elif value < 0.8:
            return f"🚨 BIAS DETECTED ({value:.3f}) - Unprivileged group disadvantaged"
        else:
            return f"🚨 BIAS DETECTED ({value:.3f}) - Privileged group disadvantaged"
    
    elif metric_name in ['equal_opportunity_diff', 'demographic_parity_diff']:
        if value is None:
            return "⚠️ Could not compute (insufficient data)"
        elif abs(value) <= 0.1:
            return f"✅ FAIR ({value:+.3f}) - Difference within ±10%"
        elif value < -0.1:
            return f"🚨 BIAS ({value:+.3f}) - Unprivileged group disadvantaged"
        else:
            return f"🚨 BIAS ({value:+.3f}) - Privileged group disadvantaged"
    
    return "Unknown metric"

def plot_fairness_comparison(fairness_results: Dict, save_path: str = None):
    """Plot approval rates by group."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    groups = list(fairness_results['group_metrics'].keys())
    approval_rates = [fairness_results['group_metrics'][g]['approval_rate'] for g in groups]
    counts = [fairness_results['group_metrics'][g]['count'] for g in groups]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(groups)]
    bars = ax.bar(groups, approval_rates, color=colors, edgecolor='black')
    
    ax.set_ylabel('Approval Rate', fontsize=12)
    ax.set_xlabel(fairness_results['attribute'], fontsize=12)
    ax.set_title(f"Approval Rate by {fairness_results['attribute']}", 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    
    for bar, rate, count in zip(bars, approval_rates, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.1%}\\n(n={count})', ha='center', fontweight='bold')
    
    ax.axhline(y=np.mean(approval_rates), color='red', linestyle='--', 
               label=f'Overall Rate: {np.mean(approval_rates):.1%}')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_fairness_report(all_results: List[Dict]) -> str:
    """Generate comprehensive fairness analysis report."""
    report = []
    report.append("\\n" + "=" * 70)
    report.append("🔍 FAIRNESS & BIAS ANALYSIS REPORT")
    report.append("=" * 70)
    
    for result in all_results:
        attr = result['attribute']
        report.append(f"\\n📊 Protected Attribute: {attr}")
        report.append("-" * 50)
        report.append(f"   Privileged Group: {result['privileged_group']}")
        report.append(f"   Unprivileged Group: {result['unprivileged_group']}")
        
        report.append("\\n   Group Statistics:")
        for group, metrics in result['group_metrics'].items():
            report.append(f"     • {group}: n={metrics['count']}, "
                         f"Approval Rate={metrics['approval_rate']:.1%}, "
                         f"Precision={metrics['precision']:.3f}, "
                         f"Recall={metrics['recall']:.3f}")
        
        report.append("\\n   Fairness Metrics:")
        report.append(f"     • Disparate Impact: "
                     f"{interpret_fairness_metric('disparate_impact', result['disparate_impact'])}")
        report.append(f"     • Equal Opportunity Diff: "
                     f"{interpret_fairness_metric('equal_opportunity_diff', result['equal_opportunity_diff'])}")
        report.append(f"     • Demographic Parity Diff: "
                     f"{interpret_fairness_metric('demographic_parity_diff', result['demographic_parity_diff'])}")
    
    report.append("\\n" + "=" * 70)
    report.append("📋 RECOMMENDATIONS")
    report.append("=" * 70)
    report.append("""
1. If Disparate Impact < 0.8:
   - Review feature importance for protected attributes
   - Consider removing or reweighting biased features
   - Implement fairness constraints during training

2. If Equal Opportunity Difference is significant:
   - Analyze false negative rates across groups
   - Consider threshold adjustment for underperforming groups
   - Use rejection inference for bias correction

3. General Best Practices:
   - Collect more representative data
   - Use fairness-aware algorithms (e.g., Fairlearn)
   - Implement continuous monitoring for bias drift
   - Document fairness metrics in model cards
""")
    
    return "\\n".join(report)

def run_full_fairness_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                                y_prob: np.ndarray, df: pd.DataFrame,
                                save_dir: str = None) -> List[Dict]:
    """Run fairness analysis on all protected attributes."""
    print("\\n🔍 RUNNING FAIRNESS ANALYSIS")
    print("=" * 50)
    
    protected_attrs = {
        'Gender': ('Male', 'Female'),
        'Married': ('Yes', 'No'),
        'Education': ('Graduate', 'Not Graduate'),
        'Property_Area': ('Urban', 'Rural')
    }
    
    all_results = []
    
    for attr, (priv, unpriv) in protected_attrs.items():
        if attr not in df.columns:
            continue
        
        print(f"\\n  Analyzing: {attr}...")
        
        result = analyze_fairness_for_attribute(
            y_true, y_pred, y_prob,
            df[attr].values,
            attr, priv, unpriv
        )
        all_results.append(result)
        
        if save_dir:
            plot_fairness_comparison(result, f"{save_dir}/fairness_{attr.lower()}.png")
        else:
            plot_fairness_comparison(result)
    
    # Print report
    report = generate_fairness_report(all_results)
    print(report)
    
    return all_results
