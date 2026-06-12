import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.features import build_preprocessing_pipeline

def create_logistic_regression_pipeline(high_card_cols, low_card_cols, numerical_cols, class_weight='balanced'):
    """
    Standard pipeline: TargetEncoder + Standard Scaler + Logistic Regression
    """
    preprocessor = build_preprocessing_pipeline(high_card_cols, low_card_cols, numerical_cols)
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=42
        ))
    ])
    
    return pipeline

def create_smote_rf_pipeline(high_card_cols, low_card_cols, numerical_cols):
    """
    Oversampling pipeline: TargetEncoder + SMOTE + Random Forest Classifier
    """
    preprocessor = build_preprocessing_pipeline(high_card_cols, low_card_cols, numerical_cols)
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(sampling_strategy=0.1, random_state=42)), # oversample minority to 10%
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return pipeline

def create_cost_sensitive_rf_pipeline(high_card_cols, low_card_cols, numerical_cols):
    """
    Cost-sensitive Random Forest pipeline using class weights.
    """
    preprocessor = build_preprocessing_pipeline(high_card_cols, low_card_cols, numerical_cols)
    
    # We assign higher weights to class 1 to penalize False Negatives
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight={0: 1.0, 1: 25.0}, # Class 1 is 25x more important
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return pipeline
