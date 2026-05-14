import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Ensure src directory is in path for robust importing
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from database import fetch_all_data

def preprocess_data():
    """
    Preprocesses the merged dataset for Machine Learning:
    1. Handles missing values cleanly.
    2. Drops non-predictive or leaky features (customer_id, churn_reason, churn_date).
    3. Encodes categorical columns using LabelEncoder.
    4. Scales numerical columns using StandardScaler.
    5. Splits data into train and test sets (80/20).
    
    Returns:
        X_train_scaled (pd.DataFrame): Scaled training features with column names preserved.
        X_test_scaled (pd.DataFrame): Scaled testing features with column names preserved.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Testing target labels.
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING FOR MACHINE LEARNING")
    print("="*60)
    
    df = fetch_all_data()
    if df.empty:
        raise ValueError("[Error] No data fetched for preprocessing.")
        
    print(f"Original dataset shape: {df.shape}")
    
    # 1. Handle missing values
    # Convert total_charges safely to numeric
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    
    missing_before = df['total_charges'].isnull().sum()
    if missing_before > 0:
        print(f"Handling {missing_before} missing values in 'total_charges' using median imputation...")
        median_val = df['total_charges'].median()
        df['total_charges'] = df['total_charges'].fillna(median_val)
        
    print(f"Missing values remaining in features: {df[['age', 'tenure', 'monthly_charges', 'total_charges']].isnull().sum().sum()}")
    
    # 2. Separate target and drop identifiers/leakage columns
    # Professional Insight: churn_reason and churn_date only exist AFTER a customer churns.
    # Keeping them in the feature set would cause critical data leakage.
    leaky_cols = ['customer_id', 'churn_reason', 'churn_date']
    print(f"Dropping identifier and leaky columns to prevent data leakage: {leaky_cols}")
    
    X = df.drop(columns=leaky_cols + ['churned'])
    y = df['churned']
    
    # Encode Target column using LabelEncoder (Yes -> 1, No -> 0)
    target_encoder = LabelEncoder()
    y_encoded = pd.Series(target_encoder.fit_transform(y), name='churned')
    print(f"Target 'churned' encoded successfully: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
    
    # 3. Encode categorical columns using LabelEncoder as requested
    categorical_cols = ['gender', 'contract_type', 'phone_service', 'internet_service', 'streaming_tv', 'streaming_movies']
    print(f"Encoding categorical columns using LabelEncoder: {categorical_cols}")
    
    # Keep a copy of feature names
    X_encoded = X.copy()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le
        
    # 4. Split data into train and test sets (80/20) BEFORE scaling to prevent data leakage
    print("Splitting data into train and test sets (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print(f"Train set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 5. Scale numerical columns using StandardScaler
    numerical_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
    print(f"Scaling numerical columns using StandardScaler: {numerical_cols}")
    
    scaler = StandardScaler()
    
    # Fit scaler on training data only, then transform both train and test sets
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print("Preprocessing completed successfully!")
    print("="*60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Quick standalone test
    try:
        X_tr, X_te, y_tr, y_te = preprocess_data()
        print("\nSample Preprocessed Training Features (head):")
        print(X_tr.head().to_string())
    except Exception as e:
        print(f"Error during preprocessing standalone test: {e}")
