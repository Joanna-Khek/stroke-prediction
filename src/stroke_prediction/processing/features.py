import numpy as np
import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin

class Mapper(BaseEstimator, TransformerMixin):
    """Map feature to value in dictionary"""

    def __init__(self, feature: List[str], mappings: dict):
        self.feature = feature
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
    
    def transform(self, X: pd.DataFrame):
        X = X.copy()

        for feature in self.feature:
            X[feature] = X[feature].map(self.mappings)

        return X