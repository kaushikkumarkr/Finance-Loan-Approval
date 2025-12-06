"""
Loan Approval Prediction System - Model Training Module
========================================================
Complete model training pipeline with multiple classifiers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

def get_base_models() -> Dict[str, Any]:
    """Get dictionary of base models with default parameters."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0),
        'SVM': SVC(probability=True, random_state=42)
    }

def train_base_models(X_train: pd.DataFrame, y_train: pd.Series, 
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Train all base models and return results."""
    models = get_base_models()
    results = {}
    
    print("\\n🚀 TRAINING BASE MODELS")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"  Training {name}...", end=" ")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        print("✅")
    
    return results

def cross_validate_models(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> pd.DataFrame:
    """Perform stratified cross-validation for all models."""
    models = get_base_models()
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    cv_results = []
    
    print(f"\\n🔄 CROSS-VALIDATION ({cv}-Fold Stratified)")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"  CV for {name}...", end=" ")
        
        scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        
        cv_results.append({
            'Model': name,
            'Mean AUC': scores.mean(),
            'Std AUC': scores.std(),
            'Min AUC': scores.min(),
            'Max AUC': scores.max()
        })
        
        print(f"AUC: {scores.mean():.4f} (±{scores.std():.4f})")
    
    return pd.DataFrame(cv_results).sort_values('Mean AUC', ascending=False)

class OptunaObjective:
    """Optuna objective for hyperparameter tuning."""
    
    def __init__(self, X, y, model_name):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    def __call__(self, trial):
        if self.model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
        
        elif self.model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'verbosity': 0
            }
            model = XGBClassifier(**params)
        
        elif self.model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'random_state': 42,
                'verbose': -1
            }
            model = LGBMClassifier(**params)
        
        elif self.model_name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_state': 42,
                'verbose': 0
            }
            model = CatBoostClassifier(**params)
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        score = cross_val_score(model, self.X, self.y, cv=self.skf, scoring='roc_auc').mean()
        return score

def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, 
                         model_name: str, n_trials: int = 50) -> Tuple[Dict, float]:
    """Tune hyperparameters using Optuna."""
    print(f"\\n🔧 HYPERPARAMETER TUNING: {model_name}")
    print("=" * 50)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    objective = OptunaObjective(X, y, model_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\\n  Best AUC: {study.best_value:.4f}")
    print(f"  Best Parameters: {study.best_params}")
    
    return study.best_params, study.best_value

def train_tuned_model(X_train: pd.DataFrame, y_train: pd.Series,
                      model_name: str, params: Dict) -> Any:
    """Train a model with tuned hyperparameters."""
    if model_name == 'Random Forest':
        model = RandomForestClassifier(**params, random_state=42)
    elif model_name == 'XGBoost':
        model = XGBClassifier(**params, random_state=42, verbosity=0)
    elif model_name == 'LightGBM':
        model = LGBMClassifier(**params, random_state=42, verbose=-1)
    elif model_name == 'CatBoost':
        model = CatBoostClassifier(**params, random_state=42, verbose=0)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.fit(X_train, y_train)
    print(f"\\n✅ Trained {model_name} with optimized parameters")
    
    return model
