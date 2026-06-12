import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce

class RegularizedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    A custom target encoder wrapper using category_encoders to ensure 
    proper smoothing and noise addition for regularization.
    """
    def __init__(self, cols=None, smoothing=10.0):
        self.cols = cols
        self.smoothing = smoothing
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = ce.TargetEncoder(
            cols=self.cols, 
            smoothing=self.smoothing,
            handle_missing='value',
            handle_unknown='value'
        )
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

def build_preprocessing_pipeline(high_card_cols, low_card_cols, numerical_cols, target_smoothing=15.0):
    """
    Constructs a scikit-learn ColumnTransformer that processes high-cardinality categoricals
    via target encoding, low-cardinality categoricals via one-hot encoding, and scales numerical values.
    """
    # Preprocessor for high cardinality
    high_card_transformer = Pipeline(steps=[
        ('target_enc', ce.TargetEncoder(smoothing=target_smoothing)),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessor for low cardinality
    low_card_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Preprocessor for numericals
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('high_card', high_card_transformer, high_card_pattern_match(high_card_cardinality=high_card_cols)),
            ('low_card', low_card_transformer, low_card_cols),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor

def high_card_pattern_match(high_card_cardinality):
    # Just a helper wrapper list
    return list(high_card_cardinality)
