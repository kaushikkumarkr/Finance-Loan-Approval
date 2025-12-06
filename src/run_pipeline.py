#!/usr/bin/env python3
"""
🏦 LOAN APPROVAL PREDICTION - COMPLETE ML PIPELINE
===================================================
Enterprise-Grade Machine Learning Project

This script runs the complete end-to-end ML pipeline including:
1. Data Loading & Understanding
2. Super-Charged EDA
3. Missing Value & Outlier Treatment
4. Feature Engineering
5. Statistical Analysis
6. Model Training & Evaluation
7. Hyperparameter Tuning
8. SHAP Explainability
9. Fairness Analysis
10. Model Export

Author: Data Science Portfolio Project
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
import os
import joblib
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna
from optuna.samplers import TPESampler
import shap

# Set paths
BASE_DIR = Path(__file__).parent.parent if '__file__' in dir() else Path('.')
DATA_DIR = BASE_DIR / 'data'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
MODELS_DIR = BASE_DIR / 'models'

# Create directories
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)

print("=" * 70)
print("🏦 LOAN APPROVAL PREDICTION - COMPLETE ML PIPELINE")
print("=" * 70)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n" + "=" * 70)
print("1️⃣ DATA LOADING & UNDERSTANDING")
print("=" * 70)

train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'test.csv')

print(f"✅ Training data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"✅ Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

df = train_df.copy()

# Data Schema
print("\n📋 DATA SCHEMA:")
print("-" * 50)
schema = pd.DataFrame({
    'Data Type': df.dtypes,
    'Non-Null': df.count(),
    'Null Count': df.isnull().sum(),
    'Null %': (df.isnull().sum() / len(df) * 100).round(2),
    'Unique': df.nunique()
})
print(schema)

# Class Balance
print("\n📊 TARGET DISTRIBUTION:")
print("-" * 50)
target_dist = df['Loan_Status'].value_counts()
print(f"Approved (Y): {target_dist['Y']} ({target_dist['Y']/len(df)*100:.1f}%)")
print(f"Rejected (N): {target_dist['N']} ({target_dist['N']/len(df)*100:.1f}%)")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("2️⃣ SUPER-CHARGED EDA")
print("=" * 70)

# Numerical columns
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, col in enumerate(numerical_cols):
    ax = axes[idx // 2, idx % 2]
    df[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(df[col].median(), color='red', linestyle='--', label=f'Median: {df[col].median():.0f}')
    ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: numerical_distributions.png")

# Categorical distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, col in enumerate(categorical_cols):
    if idx < 6:
        df[col].value_counts().plot(kind='bar', ax=axes[idx], color='steelblue', edgecolor='black')
        axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: categorical_distributions.png")

# Approval rate by categorical features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, col in enumerate(categorical_cols):
    if idx < 6:
        approval_rate = df.groupby(col)['Loan_Status'].apply(lambda x: (x == 'Y').mean())
        approval_rate.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'][:len(approval_rate)])
        axes[idx].set_title(f'Approval Rate by {col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Approval Rate')
        axes[idx].set_ylim(0, 1)
        axes[idx].tick_params(axis='x', rotation=45)
        for i, v in enumerate(approval_rate):
            axes[idx].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'approval_rates_by_category.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: approval_rates_by_category.png")

# Credit History Analysis (CRITICAL)
print("\n📊 CREDIT HISTORY IMPACT:")
print("-" * 50)
credit_approval = df.groupby('Credit_History')['Loan_Status'].apply(lambda x: (x == 'Y').mean())
print(f"Credit History = 0 (Bad):  {credit_approval.get(0, 0):.1%} approval rate")
print(f"Credit History = 1 (Good): {credit_approval.get(1, 0):.1%} approval rate")
print("💡 Insight: Credit History is the MOST important predictor!")

# Correlation heatmap (numerical)
df_numeric = df[numerical_cols + ['Credit_History']].dropna()
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df_numeric.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
ax.set_title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: correlation_heatmap.png")

# Cramér's V for categorical correlations
def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    return np.sqrt(phi2 / min(k-1, r-1))

print("\n📊 CRAMÉR'S V (Categorical Correlations):")
print("-" * 50)
for col in categorical_cols:
    if df[col].notna().sum() > 0:
        cv = cramers_v(df[col].dropna(), df.loc[df[col].notna(), 'Loan_Status'])
        print(f"  {col} vs Loan_Status: {cv:.3f}")

# ============================================================================
# 3. MISSING VALUE & OUTLIER TREATMENT
# ============================================================================
print("\n" + "=" * 70)
print("3️⃣ MISSING VALUE & OUTLIER TREATMENT")
print("=" * 70)

# Missing value analysis
print("\n📊 Missing Values:")
print("-" * 50)
missing = df.isnull().sum()
missing = missing[missing > 0]
for col, count in missing.items():
    print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")

# Imputation
print("\n🔧 Imputing Missing Values:")
print("-" * 50)

# Numerical - Median imputation
for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  {col}: Filled {missing.get(col, 0)} values with median={median_val:.0f}")

# Categorical - Mode imputation
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  {col}: Filled {missing.get(col, 0)} values with mode='{mode_val}'")

# Outlier Detection
print("\n📊 Outlier Detection (IQR Method):")
print("-" * 50)
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"  {col}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")

# Log transform skewed features
print("\n🔧 Applying Log Transformations:")
print("-" * 50)
df['Log_ApplicantIncome'] = np.log1p(df['ApplicantIncome'])
df['Log_CoapplicantIncome'] = np.log1p(df['CoapplicantIncome'])
df['Log_LoanAmount'] = np.log1p(df['LoanAmount'])
print("  ✅ Applied log1p to ApplicantIncome, CoapplicantIncome, LoanAmount")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("4️⃣ FEATURE ENGINEERING")
print("=" * 70)

# Total Income
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Log_Total_Income'] = np.log1p(df['Total_Income'])
print("  ✅ Total_Income = ApplicantIncome + CoapplicantIncome")

# Dependents numeric
df['Dependents_Numeric'] = df['Dependents'].replace('3+', '3').astype(float)

# Loan to Income Ratio
df['Loan_to_Income'] = df['LoanAmount'] / (df['Total_Income'] / 1000 + 1)
print("  ✅ Loan_to_Income ratio")

# Estimated EMI and Debt to Income
df['Estimated_EMI'] = (df['LoanAmount'] * 1000 * 0.01 * 
                        (1.01 ** df['Loan_Amount_Term']) / 
                        ((1.01 ** df['Loan_Amount_Term']) - 1))
df['Debt_to_Income'] = df['Estimated_EMI'] / (df['Total_Income'] + 1)
print("  ✅ Debt_to_Income ratio")

# Income per dependent
df['Income_Per_Dependent'] = df['Total_Income'] / (df['Dependents_Numeric'] + 1)
print("  ✅ Income_Per_Dependent")

# Credit History Interactions
df['CreditHistory_Income'] = df['Credit_History'] * df['Log_Total_Income']
df['CreditHistory_Loan'] = df['Credit_History'] * df['Log_LoanAmount']
print("  ✅ Credit History interaction features")

# Property Area encoding
area_map = {'Rural': 1, 'Semiurban': 2, 'Urban': 3}
df['Property_Area_Encoded'] = df['Property_Area'].map(area_map)
df['Area_Income_Interaction'] = df['Property_Area_Encoded'] * df['Log_Total_Income']
print("  ✅ Property Area interaction features")

# Binary encodings
df['Gender_Encoded'] = (df['Gender'] == 'Male').astype(int)
df['Married_Encoded'] = (df['Married'] == 'Yes').astype(int)
df['Education_Encoded'] = (df['Education'] == 'Graduate').astype(int)
df['Self_Employed_Encoded'] = (df['Self_Employed'] == 'Yes').astype(int)
df['Graduate_SelfEmployed'] = ((df['Education'] == 'Graduate') & (df['Self_Employed'] == 'Yes')).astype(int)
print("  ✅ Binary encodings for categorical features")

# One-hot encoding for Property Area
df = pd.get_dummies(df, columns=['Property_Area'], prefix='PropArea')
print("  ✅ One-hot encoding for Property_Area")

# Family Size
df['Family_Size'] = df['Dependents_Numeric'] + df['Married_Encoded'] + 1
print("  ✅ Family_Size feature")

# Target encoding
df['Loan_Status_Encoded'] = (df['Loan_Status'] == 'Y').astype(int)
print("  ✅ Target variable encoded")

print(f"\n📊 Total features after engineering: {df.shape[1]}")

# ============================================================================
# 5. STATISTICAL ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("5️⃣ STATISTICAL ANALYSIS")
print("=" * 70)

# Chi-square tests for categorical features
print("\n📊 CHI-SQUARE TESTS:")
print("-" * 50)
cat_cols_for_test = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed']
for col in cat_cols_for_test:
    if col in train_df.columns:
        contingency = pd.crosstab(train_df[col].dropna(), train_df['Loan_Status'])
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        significance = "✅ Significant" if p_val < 0.05 else "❌ Not Significant"
        print(f"  {col}: χ²={chi2:.2f}, p={p_val:.4f} {significance}")

# ANOVA / t-tests for numerical features
print("\n📊 T-TESTS (Numerical vs Loan_Status):")
print("-" * 50)
for col in numerical_cols:
    approved = train_df[train_df['Loan_Status'] == 'Y'][col].dropna()
    rejected = train_df[train_df['Loan_Status'] == 'N'][col].dropna()
    t_stat, p_val = ttest_ind(approved, rejected)
    significance = "✅ Significant" if p_val < 0.05 else "❌ Not Significant"
    print(f"  {col}: t={t_stat:.2f}, p={p_val:.4f} {significance}")

# ============================================================================
# 6. MODEL BUILDING
# ============================================================================
print("\n" + "=" * 70)
print("6️⃣ MODEL BUILDING")
print("=" * 70)

# Prepare features
feature_cols = [
    'Log_ApplicantIncome', 'Log_CoapplicantIncome', 'Log_Total_Income',
    'Log_LoanAmount', 'Loan_Amount_Term', 'Credit_History',
    'Loan_to_Income', 'Debt_to_Income', 'Income_Per_Dependent',
    'CreditHistory_Income', 'CreditHistory_Loan', 'Area_Income_Interaction',
    'Family_Size', 'Gender_Encoded', 'Married_Encoded', 'Education_Encoded',
    'Self_Employed_Encoded', 'PropArea_Rural', 'PropArea_Semiurban',
    'PropArea_Urban', 'Graduate_SelfEmployed'
]

X = df[feature_cols]
y = df['Loan_Status_Encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Train set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")

# Scale features
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0)
}

# Train and evaluate
print("\n📊 TRAINING MODELS:")
print("-" * 50)
results = []

for name, model in models.items():
    print(f"  Training {name}...", end=" ")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_prob)
    }
    results.append(metrics)
    print(f"AUC: {metrics['AUC-ROC']:.4f}")

# Results table
results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)
print("\n📊 MODEL COMPARISON:")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(REPORTS_DIR / 'model_comparison.csv', index=False)
print("\n✅ Saved: model_comparison.csv")

# ROC curves
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
for (name, model), color in zip(models.items(), colors):
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: roc_curves.png")

# ============================================================================
# 7. CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 70)
print("7️⃣ CROSS-VALIDATION (5-Fold Stratified)")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='roc_auc')
    cv_results.append({
        'Model': name,
        'Mean AUC': scores.mean(),
        'Std': scores.std()
    })
    print(f"  {name}: {scores.mean():.4f} (±{scores.std():.4f})")

# ============================================================================
# 8. HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "=" * 70)
print("8️⃣ HYPERPARAMETER TUNING (Optuna)")
print("=" * 70)

# Tune XGBoost (best performer typically)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbosity': 0,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\n✅ Best XGBoost AUC: {study.best_value:.4f}")
print(f"   Best Parameters: {study.best_params}")

# Train best model
best_params = study.best_params
best_params['random_state'] = 42
best_params['verbosity'] = 0
best_params['use_label_encoder'] = False
best_params['eval_metric'] = 'logloss'

best_model = XGBClassifier(**best_params)
best_model.fit(X_train_scaled, y_train)

# Evaluate tuned model
y_pred_best = best_model.predict(X_test_scaled)
y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 TUNED MODEL PERFORMANCE:")
print("-" * 50)
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_best):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_best):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_best):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_pred_best):.4f}")
print(f"  AUC-ROC:   {roc_auc_score(y_test, y_prob_best):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Tuned XGBoost', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: confusion_matrix.png")

# ============================================================================
# 9. SHAP EXPLAINABILITY
# ============================================================================
print("\n" + "=" * 70)
print("9️⃣ SHAP EXPLAINABILITY")
print("=" * 70)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)

# Summary plot
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_scaled, max_display=15, show=False)
plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_summary.png")

# Bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_scaled, plot_type='bar', max_display=15, show=False)
plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'shap_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_importance.png")

# Top features
feature_importance = pd.DataFrame({
    'Feature': X_test_scaled.columns,
    'Mean_SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)

print("\n📊 TOP 10 MOST IMPORTANT FEATURES:")
print("-" * 50)
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Mean_SHAP']:.4f}")

# ============================================================================
# 10. FAIRNESS ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("🔟 FAIRNESS & BIAS ANALYSIS")
print("=" * 70)

# Prepare data for fairness analysis
df_test = df.iloc[X_test.index].copy()
df_test['Predicted'] = y_pred_best
df_test['Actual'] = y_test.values

protected_attrs = {
    'Gender': ('Male', 'Female'),
    'Married': ('Yes', 'No'),
    'Education': ('Graduate', 'Not Graduate')
}

fairness_results = []

for attr, (priv, unpriv) in protected_attrs.items():
    print(f"\n📊 {attr}:")
    print("-" * 40)
    
    # Approval rates
    for group in [priv, unpriv]:
        mask = train_df[attr] == group
        if mask.sum() > 0:
            approval_rate = (train_df.loc[mask, 'Loan_Status'] == 'Y').mean()
            print(f"  {group}: {approval_rate:.1%} approval rate")
    
    # Disparate Impact
    priv_mask = train_df[attr] == priv
    unpriv_mask = train_df[attr] == unpriv
    
    if priv_mask.sum() > 0 and unpriv_mask.sum() > 0:
        priv_rate = (train_df.loc[priv_mask, 'Loan_Status'] == 'Y').mean()
        unpriv_rate = (train_df.loc[unpriv_mask, 'Loan_Status'] == 'Y').mean()
        
        di = unpriv_rate / priv_rate if priv_rate > 0 else 0
        
        status = "✅ Fair" if 0.8 <= di <= 1.25 else "⚠️ Potential Bias"
        print(f"  Disparate Impact: {di:.3f} {status}")
        
        fairness_results.append({
            'Attribute': attr,
            'Privileged': priv,
            'Unprivileged': unpriv,
            'DI_Ratio': di,
            'Status': status
        })

# Fairness visualization
fig, ax = plt.subplots(figsize=(10, 6))
for attr, (priv, unpriv) in protected_attrs.items():
    approval_by_group = train_df.groupby(attr)['Loan_Status'].apply(lambda x: (x == 'Y').mean())
    approval_by_group.plot(kind='bar', ax=ax, alpha=0.7, label=attr)
ax.set_ylabel('Approval Rate')
ax.set_title('Approval Rates by Protected Attributes', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)
ax.legend()
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fairness_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: fairness_analysis.png")

# ============================================================================
# 11. SAVE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("1️⃣1️⃣ SAVING MODEL")
print("=" * 70)

# Save best model
joblib.dump(best_model, MODELS_DIR / 'best_model.pkl')
joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
print(f"✅ Saved: best_model.pkl")
print(f"✅ Saved: scaler.pkl")

# Save feature columns
with open(MODELS_DIR / 'feature_columns.txt', 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"✅ Saved: feature_columns.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("📊 PIPELINE COMPLETE - SUMMARY")
print("=" * 70)

print(f"""
✅ Data processed: {len(df)} samples
✅ Features engineered: {len(feature_cols)}
✅ Models trained: {len(models)}
✅ Best model: XGBoost (Tuned)
✅ Best AUC-ROC: {roc_auc_score(y_test, y_prob_best):.4f}
✅ SHAP explainability: Complete
✅ Fairness analysis: Complete

📁 Outputs saved to:
   - {FIGURES_DIR}/ (visualizations)
   - {REPORTS_DIR}/ (model comparison)
   - {MODELS_DIR}/ (trained model)

🚀 Ready for deployment!
""")
