"""
Loan Approval Prediction System - Feature Engineering Module
=============================================================
Industry-grade feature engineering for credit risk modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from typing import Tuple, List

def create_income_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create income-related features."""
    df_fe = df.copy()
    
    # Total Income
    df_fe['Total_Income'] = df_fe['ApplicantIncome'] + df_fe['CoapplicantIncome']
    print("  ✅ Created Total_Income = ApplicantIncome + CoapplicantIncome")
    
    # Income per dependent
    df_fe['Dependents_Numeric'] = df_fe['Dependents'].replace('3+', '3').astype(float)
    df_fe['Income_Per_Dependent'] = df_fe['Total_Income'] / (df_fe['Dependents_Numeric'] + 1)
    print("  ✅ Created Income_Per_Dependent")
    
    # Income bins
    df_fe['Income_Bin'] = pd.qcut(df_fe['Total_Income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    print("  ✅ Created Income_Bin (quartile-based)")
    
    # Log transformations for skewed distributions
    df_fe['Log_ApplicantIncome'] = np.log1p(df_fe['ApplicantIncome'])
    df_fe['Log_CoapplicantIncome'] = np.log1p(df_fe['CoapplicantIncome'])
    df_fe['Log_Total_Income'] = np.log1p(df_fe['Total_Income'])
    print("  ✅ Created log-transformed income features")
    
    return df_fe

def create_loan_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create loan-related features."""
    df_fe = df.copy()
    
    # Loan to Income Ratio (critical for affordability)
    df_fe['Loan_to_Income'] = df_fe['LoanAmount'] / (df_fe['Total_Income'] / 1000 + 1)
    print("  ✅ Created Loan_to_Income ratio")
    
    # Debt to Income proxy (EMI estimation)
    # Assuming average interest rate of 10% and standard EMI calculation
    df_fe['Estimated_EMI'] = df_fe['LoanAmount'] * 1000 * 0.01 * (1.01 ** df_fe['Loan_Amount_Term']) / ((1.01 ** df_fe['Loan_Amount_Term']) - 1)
    df_fe['Debt_to_Income'] = df_fe['Estimated_EMI'] / (df_fe['Total_Income'] + 1)
    print("  ✅ Created Debt_to_Income ratio")
    
    # Loan Amount bins
    df_fe['LoanAmount_Bin'] = pd.qcut(df_fe['LoanAmount'], q=4, labels=['Small', 'Medium', 'Large', 'Very Large'], duplicates='drop')
    print("  ✅ Created LoanAmount_Bin (quartile-based)")
    
    # Log transformation
    df_fe['Log_LoanAmount'] = np.log1p(df_fe['LoanAmount'])
    print("  ✅ Created Log_LoanAmount")
    
    # Loan term categories
    df_fe['Term_Category'] = df_fe['Loan_Amount_Term'].apply(
        lambda x: 'Short' if x <= 180 else ('Medium' if x <= 300 else 'Long')
    )
    print("  ✅ Created Term_Category")
    
    return df_fe

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between key variables."""
    df_fe = df.copy()
    
    # Credit History interactions (most important feature)
    df_fe['CreditHistory_Income'] = df_fe['Credit_History'] * df_fe['Log_Total_Income']
    df_fe['CreditHistory_Loan'] = df_fe['Credit_History'] * df_fe['Log_LoanAmount']
    print("  ✅ Created Credit History interaction features")
    
    # Property area & income interaction
    area_map = {'Rural': 1, 'Semiurban': 2, 'Urban': 3}
    df_fe['Property_Area_Encoded'] = df_fe['Property_Area'].map(area_map)
    df_fe['Area_Income_Interaction'] = df_fe['Property_Area_Encoded'] * df_fe['Log_Total_Income']
    print("  ✅ Created Property Area interaction features")
    
    # Education & Employment interaction
    df_fe['Graduate_SelfEmployed'] = ((df_fe['Education'] == 'Graduate') & 
                                       (df_fe['Self_Employed'] == 'Yes')).astype(int)
    print("  ✅ Created Graduate_SelfEmployed indicator")
    
    # Family size proxy
    df_fe['Family_Size'] = df_fe['Dependents_Numeric'] + (df_fe['Married'] == 'Yes').astype(int) + 1
    print("  ✅ Created Family_Size feature")
    
    return df_fe

def encode_categoricals(df: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical variables.
    Uses OneHot for low cardinality, Label encoding for ordinal.
    """
    df_encoded = df.copy()
    encoders = {}
    
    # Binary encoding
    binary_maps = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0}
    }
    
    for col, mapping in binary_maps.items():
        if col in df_encoded.columns:
            df_encoded[col + '_Encoded'] = df_encoded[col].map(mapping)
            encoders[col] = mapping
            print(f"  ✅ Binary encoded {col}")
    
    # One-hot encoding for Property_Area
    if 'Property_Area' in df_encoded.columns:
        dummies = pd.get_dummies(df_encoded['Property_Area'], prefix='PropArea')
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        print("  ✅ One-hot encoded Property_Area")
    
    # Encode target if present
    if target_col and target_col in df_encoded.columns:
        df_encoded[target_col + '_Encoded'] = (df_encoded[target_col] == 'Y').astype(int)
        print(f"  ✅ Encoded target column {target_col}")
    
    return df_encoded, encoders

def scale_features(df: pd.DataFrame, columns: List[str], method: str = 'robust') -> Tuple[pd.DataFrame, object]:
    """
    Scale numerical features.
    
    Methods:
    - 'standard': StandardScaler (mean=0, std=1)
    - 'robust': RobustScaler (uses median and IQR, robust to outliers)
    """
    df_scaled = df.copy()
    
    if method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    print(f"  ✅ Applied {method} scaling to {len(columns)} features")
    
    return df_scaled, scaler

def get_feature_columns() -> dict:
    """Return feature column groupings for modeling."""
    return {
        'numerical': [
            'Log_ApplicantIncome', 'Log_CoapplicantIncome', 'Log_Total_Income',
            'Log_LoanAmount', 'Loan_Amount_Term', 'Credit_History',
            'Loan_to_Income', 'Debt_to_Income', 'Income_Per_Dependent',
            'CreditHistory_Income', 'CreditHistory_Loan', 'Area_Income_Interaction',
            'Family_Size'
        ],
        'categorical_encoded': [
            'Gender_Encoded', 'Married_Encoded', 'Education_Encoded', 
            'Self_Employed_Encoded', 'PropArea_Rural', 'PropArea_Semiurban', 
            'PropArea_Urban', 'Graduate_SelfEmployed'
        ],
        'target': 'Loan_Status_Encoded'
    }

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run complete feature engineering pipeline."""
    print("\\n🔧 FEATURE ENGINEERING PIPELINE")
    print("=" * 50)
    
    print("\\n1️⃣ Creating Income Features...")
    df = create_income_features(df)
    
    print("\\n2️⃣ Creating Loan Features...")
    df = create_loan_features(df)
    
    print("\\n3️⃣ Creating Interaction Features...")
    df = create_interaction_features(df)
    
    print("\\n4️⃣ Encoding Categorical Variables...")
    df, encoders = encode_categoricals(df, target_col='Loan_Status')
    
    print("\\n✅ Feature Engineering Complete!")
    print(f"   Total features: {df.shape[1]}")
    
    return df
