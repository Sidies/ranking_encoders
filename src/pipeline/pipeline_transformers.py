#imports
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from tabulate import tabulate

class DebugTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that prints the data it receives.
    """
    def __init__(self, verbose=0):
        self.verbose = verbose
        
    def transform(self, X):
        if self.verbose > 0:
            if hasattr(X, 'toarray'):  # check if X is a sparse matrix
                df = pd.DataFrame(X.toarray())
                #print(tabulate(df.values, headers=df.columns, tablefmt='simple'))
                print(df.value_counts())
            else:
                df = pd.DataFrame(X)
                print(df.value_counts())
                #print(tabulate(df.values, headers=df.columns, tablefmt='simple'))
            
        return X

    def fit(self, X, y=None, **fit_params):
        return self
    
class TextPrinter(BaseEstimator, TransformerMixin):
    """
    Transformer that prints the text it receives.
    """
    def __init__(self, text, verbose=0):
        self.text = text
        self.verbose = verbose
        
    def transform(self, X):
        if self.verbose > 0:
            print(f"{self.text}")
        return X

    def fit(self, X, y=None):
        if self.verbose > 0:
            print(f"{self.text}")
        return self
