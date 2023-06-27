"""
Evaluation script to evaluate the regression task.

Example usage to evaluate a pipeline:

import pandas as pd

from category_encoders import OneHotEncoder
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.load_datasets import load_dataset

factors = ["dataset", "model", "tuning", "scoring"]
new_index = "encoder"

# ---- load data...
DATA_DIR = Path(".\data")

df_train = load_dataset(DATA_DIR / "dataset_train.csv")
X_train = df_train.drop("cv_score", axis=1)
y_train = df_train["cv_score"]
df_test = load_dataset(DATA_DIR / "dataset_test.csv")  # you do not have this dataset
X_test = df_test.drop("cv_score", axis=1)
y_test = df_test["cv_score"]

# ---- ... or split a dataset
# df_train = load_dataset(DATA_DIR / "dataset_train.csv")
# X_train, X_test, y_train, y_test = custom_train_test_split(df, factors, target)

# ---- predict
dummy_pipe = Pipeline([("encoder", OneHotEncoder()), ("model", DecisionTreeRegressor())])
y_pred = pd.Series(dummy_pipe.fit(X_train, y_train).predict(X_test), index=y_test.index, name="cv_score_pred")
df_pred = pd.concat([X_test, y_test, y_pred], axis=1)

# ---- convert to rankings and evaluate
rankings_test = get_rankings(df_pred, factors=factors, new_index=new_index, target="cv_score")
rankings_pred = get_rankings(df_pred, factors=factors, new_index=new_index, target="cv_score_pred")
print(average_spearman(rankings_test, rankings_pred))
"""

import numpy as np
import pandas as pd
import warnings 
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.model_selection import train_test_split
from typing import Iterable
from sklearn.pipeline import Pipeline
from tqdm import tqdm


def custom_cross_validation(pipeline: Pipeline, df, factors, target, train_size=0.75, shuffle=True, cv=5, verbose=1):
    """
    Like cross_validation, but keeps encoders together: the split is on 'factors'.
    """
    # check if tqdm should be disabled
    disable_tqdm = (verbose == 0)
    validation_performance_scores = {}
    for i in tqdm(range(cv), disable=disable_tqdm):
        X_train, X_test, y_train, y_test = custom_train_test_split(df, factors, target, train_size=train_size, shuffle=shuffle, random_state=i)
        
        new_index = "encoder"
        y_pred = pd.Series(pipeline.fit(X_train, y_train).predict(X_test), index=y_test.index, name="cv_score_pred")
        df_pred = pd.concat([X_test, y_test, y_pred], axis=1)        
        
        # ---- convert to rankings and evaluate
        rankings_test = get_rankings(df_pred, factors=factors, new_index=new_index, target=target)
        rankings_pred = get_rankings(df_pred, factors=factors, new_index=new_index, target=target + "_pred")
        validation_performance_scores['validation_average_spearman_fold_' + str(i)] = average_spearman(rankings_test, rankings_pred)
              
    return validation_performance_scores  
        

def custom_train_test_split(df: pd.DataFrame, factors, target, train_size=0.75, shuffle=True, random_state=0):
    """
    Like train_test_split, but keeps encoders together: the split is on 'factors'.
    """

    X_factors = df.groupby(factors).agg(lambda a: np.nan).reset_index()[factors]
    #print(f"X_faftors: \n {X_factors.head()}")
    factors_train, factors_test = train_test_split(X_factors, stratify=X_factors.dataset,
                                                   train_size=train_size, shuffle=shuffle, random_state=random_state)

    #print(f"Factors train: \n {factors_train.head()}")
    #print(f"factors test: \n {factors_test.head()}")
    
    df_train = pd.merge(factors_train, df, on=factors)
    df_test = pd.merge(factors_test, df, on=factors)
    
    #print(f"Df train:\n {df_train.head()}")
    #print(f"Df test:\n {df_test.head()}")

    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    X_test = df_test.drop(target, axis=1)
    y_test = df_test[target]

    return X_train, X_test, y_train, y_test


def score2ranking(score: pd.Series, ascending=True):
    """
    Ascending =
        True: lower score = better rank (for instance, if score is the result of a loss function or a ranking itself)
        False: greater score = better rank (for instance, if score is the result of a score such as roc_auc_score)
    """
    c = 1 if ascending else -1
    order_map = {
        s: sorted(score.unique(), key=lambda x: c * x).index(s) for s in score.unique()
    }
    return score.map(order_map)


def get_rankings(df: pd.DataFrame, factors, new_index, target) -> pd.DataFrame:

    assert set(factors).issubset(df.columns)

    rankings = {}
    for group, indices in df.groupby(factors).groups.items():
        score = df.iloc[indices].set_index(new_index)[target]
        rankings[group] = score2ranking(score, ascending=False)

    return pd.DataFrame(rankings)


def spearman_rho(x: Iterable, y: Iterable, nan_policy="omit"):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        return spearmanr(x, y, nan_policy=nan_policy)[0]


def list_spearman(rf1: pd.DataFrame, rf2: pd.DataFrame) -> np.array:
        
    if not rf1.columns.equals(rf2.columns) or not rf1.index.equals(rf2.index):
         raise ValueError("The two input dataframes should have the same index and columns.")

    return np.array([
        spearman_rho(r1, r2, nan_policy="omit") for (_, r1), (_, r2) in zip(rf1.items(), rf2.items())
    ])


def average_spearman(rf1: pd.DataFrame, rf2: pd.DataFrame) -> np.array:
    return np.nanmean(list_spearman(rf1, rf2))
