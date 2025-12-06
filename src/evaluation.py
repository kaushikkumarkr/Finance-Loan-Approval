"""
Loan Approval Prediction System - Evaluation Module
====================================================
Comprehensive model evaluation with all metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix, 
    classification_report, precision_recall_curve
)

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                        y_prob: np.ndarray = None) -> Dict:
    """Compute comprehensive evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
    
    return metrics

def evaluate_all_models(results: Dict, y_test: np.ndarray) -> pd.DataFrame:
    """Evaluate all models and create comparison table."""
    evaluation = []
    
    for name, result in results.items():
        metrics = compute_all_metrics(
            y_test, 
            result['predictions'],
            result['probabilities']
        )
        metrics['Model'] = name
        evaluation.append(metrics)
    
    df = pd.DataFrame(evaluation)
    df = df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']]
    df = df.sort_values('AUC-ROC', ascending=False)
    
    return df

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          model_name: str, save_path: str = None):
    """Plot confusion matrix with metrics."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'],
                ax=ax, annot_kws={'size': 16})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    metrics_text = f'TN={tn} | FP={fp}\\nFN={fn} | TP={tp}'
    ax.text(2.5, 0.5, metrics_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(results: Dict, y_test: np.ndarray, save_path: str = None):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for (name, result), color in zip(results.items(), colors):
        if result['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curves(results: Dict, y_test: np.ndarray, save_path: str = None):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for (name, result), color in zip(results.items(), colors):
        if result['probabilities'] is not None:
            precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
            ax.plot(recall, precision, color=color, lw=2, label=name)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_comparison(eval_df: pd.DataFrame, save_path: str = None):
    """Plot bar chart comparing all metrics across models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(eval_df)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.barh(eval_df['Model'], eval_df[metric], color=colors)
        ax.set_xlabel(metric)
        ax.set_title(metric, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.bar_label(bars, fmt='%.3f', padding=3)
    
    # Hide the last subplot
    axes[5].axis('off')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Print detailed classification report."""
    print(f"\\n📊 CLASSIFICATION REPORT: {model_name}")
    print("=" * 60)
    print(classification_report(y_true, y_pred, 
                                target_names=['Rejected', 'Approved']))

def create_summary_table(eval_df: pd.DataFrame) -> str:
    """Create a formatted summary table for the report."""
    # Rank models
    eval_df = eval_df.copy()
    eval_df['Rank'] = range(1, len(eval_df) + 1)
    
    # Format percentages
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
        eval_df[col] = eval_df[col].apply(lambda x: f'{x:.4f}')
    
    return eval_df.to_markdown(index=False)
