import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

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
                #print(tabulate(df.columns, headers=df.columns, tablefmt='simple'))
                print(df.columns)
            else:
                df = pd.DataFrame(X)
                #print(tabulate(df.columns, headers=df.columns, tablefmt='simple'))
                print(df.columns)
            
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


class ColumnKeeper(BaseEstimator, TransformerMixin):
    """
    Transformer that keeps only the specified columns.
    """
    def __init__(self, columns):
        self.columns = columns
        
    def transform(self, X):
        return X[self.columns]

    def fit(self, X, y=None):
        return self


# Custom Transformer that drops columns passed as argument
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assuming X is a DataFrame
        return X.drop(self.cols_to_drop, axis=1)
    