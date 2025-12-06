"""
Loan Approval Prediction System - Data Preprocessing Module
============================================================
Enterprise-grade data preprocessing with domain-aware imputation and outlier handling.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(train_path: str, test_path: str = None):
    """Load training and optionally test data."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    
    print(f"✅ Training data loaded: {train_df.shape}")
    if test_df is not None:
        print(f"✅ Test data loaded: {test_df.shape}")
    
    return train_df, test_df

def analyze_missing(df: pd.DataFrame, save_path: str = None):
    """Analyze and visualize missing values."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Feature': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        'Missing %', ascending=False
    )
    
    if len(missing_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(missing_df)))
        bars = ax.barh(missing_df['Feature'], missing_df['Missing %'], color=colors)
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
        ax.bar_label(bars, fmt='%.1f%%', padding=3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    return missing_df

def impute_missing(df: pd.DataFrame, strategy: str = 'domain'):
    """
    Impute missing values using domain-aware strategies.
    
    Strategies:
    - Numerical: Median (robust to outliers)
    - Categorical: Mode (most frequent)
    - Credit_History: Mode (critical feature)
    """
    df_imputed = df.copy()
    
    # Numerical columns
    numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in numerical_cols:
        if df_imputed[col].isnull().sum() > 0:
            median_val = df_imputed[col].median()
            df_imputed[col].fillna(median_val, inplace=True)
            print(f"  📊 {col}: Imputed {df[col].isnull().sum()} values with median={median_val:.2f}")
    
    # Categorical columns
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
    for col in categorical_cols:
        if df_imputed[col].isnull().sum() > 0:
            mode_val = df_imputed[col].mode()[0]
            df_imputed[col].fillna(mode_val, inplace=True)
            print(f"  📊 {col}: Imputed {df[col].isnull().sum()} values with mode='{mode_val}'")
    
    return df_imputed

def detect_outliers_iqr(df: pd.DataFrame, columns: list):
    """Detect outliers using IQR method."""
    outlier_info = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'lower_bound': lower,
            'upper_bound': upper
        }
    
    return outlier_info

def detect_outliers_zscore(df: pd.DataFrame, columns: list, threshold: float = 3.0):
    """Detect outliers using Z-score method."""
    outlier_info = {}
    
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = z_scores > threshold
        outlier_info[col] = {
            'count': outliers.sum(),
            'percentage': outliers.sum() / len(df) * 100
        }
    
    return outlier_info

def detect_outliers_isolation_forest(df: pd.DataFrame, columns: list, contamination: float = 0.1):
    """Detect outliers using Isolation Forest."""
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    data = df[columns].dropna()
    predictions = iso_forest.fit_predict(data)
    outliers = predictions == -1
    
    return {
        'total_outliers': outliers.sum(),
        'percentage': outliers.sum() / len(data) * 100
    }

def handle_outliers(df: pd.DataFrame, columns: list, method: str = 'cap'):
    """
    Handle outliers using specified method.
    
    Methods:
    - 'cap': Cap at IQR bounds
    - 'log': Log transform
    - 'remove': Remove outliers
    """
    df_handled = df.copy()
    
    for col in columns:
        if method == 'cap':
            Q1 = df_handled[col].quantile(0.25)
            Q3 = df_handled[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_handled[col] = df_handled[col].clip(lower, upper)
            print(f"  📊 {col}: Capped at [{lower:.2f}, {upper:.2f}]")
        
        elif method == 'log':
            df_handled[col] = np.log1p(df_handled[col])
            print(f"  📊 {col}: Applied log1p transformation")
    
    return df_handled

def visualize_outliers(df: pd.DataFrame, columns: list, save_path: str = None):
    """Visualize outliers using boxplots."""
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    if n_cols == 1:
        axes = [axes]
    
    for ax, col in zip(axes, columns):
        sns.boxplot(data=df, y=col, ax=ax, color='steelblue')
        ax.set_title(f'{col} Distribution', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
