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
target = "cv_score"

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

# ---- predict ...
dummy_pipe = Pipeline([("encoder", OneHotEncoder()), ("model", DecisionTreeRegressor())])
y_pred = pd.Series(dummy_pipe.fit(X_train, y_train).predict(X_test), index=y_test.index, name="cv_score_pred")
df_pred = pd.concat([X_test, y_test, y_pred], axis=1)

# ---- ... or tune
dummy_pipe = Pipeline([("encoder", OneHotEncoder()), ("model", DecisionTreeRegressor())])
indices = er.custom_cross_validated_indices(df_train, factors, target, n_splits=5, shuffle=True, random_state=1444)

gs = GridSearchCV(dummy_pipe, {"model__max_depth": [2, 5, None]}, cv=indices).fit(X_train, y_train)
print(gs.best_params_)

# ---- convert to rankings and evaluate
rankings_test = get_rankings(df_pred, factors=factors, new_index=new_index, target="cv_score")
rankings_pred = get_rankings(df_pred, factors=factors, new_index=new_index, target="cv_score_pred")
print(average_spearman(rankings_test, rankings_pred))
"""

import numpy as np
import pandas as pd
import warnings

from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics._scorer import _PredictScorer
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from typing import Iterable, List

from src.features.encoder_utils import NoY



def custom_cross_validated_indices(
        df: pd.DataFrame,
        factors: Iterable[str],
        target: str,
        **kfoldargs
) -> List[List[Iterable[int]]]:
    df_factors = df.groupby(factors)[target].mean().reset_index()
    X_factors, y_factors = df_factors.drop(target, axis=1), df_factors[target]

    indices = []
    for itr, ite in KFold(**kfoldargs).split(X_factors, y_factors):
        tr = pd.merge(X_factors.iloc[itr], df.reset_index(), on=factors)["index"]  # "index" is the index of df
        te = pd.merge(X_factors.iloc[ite], df.reset_index(), on=factors)["index"]  # "index" is the index of df
        indices.append([tr, te])

    return indices
        

def custom_train_test_split(df: pd.DataFrame, factors, target, train_size=0.75, shuffle=True, random_state=0):
    """
    Like train_test_split, but keeps encoders together: the split is on 'factors'.
    """

    X_factors = df.groupby(factors).agg(lambda a: np.nan).reset_index()[factors]
    #print(f"X_faftors: \n {X_factors.head()}")
    factors_train, factors_test = train_test_split(
        X_factors,
        stratify=X_factors.dataset,
        train_size=train_size,
        shuffle=shuffle,
        random_state=random_state
    )

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
        # score = df.iloc[indices].set_index(new_index)[target]
        score = df.loc[indices].set_index(new_index)[target]
        rankings[group] = score2ranking(score, ascending=False)

    return pd.DataFrame(rankings)


def spearman_rho(x: Iterable, y: Iterable, nan_policy="omit"):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        return spearmanr(x, y, nan_policy=nan_policy)[0]


def list_spearman(rf1: pd.DataFrame, rf2: pd.DataFrame) -> np.array:
    # if not rf1.columns.equals(rf2.columns) or not rf1.index.equals(rf2.index):
    #     raise ValueError("The two input dataframes should have the same index and columns.")

    if not rf1.index.equals(rf2.index):
        raise ValueError("The two input dataframes should have the same index and columns.")

    return np.array([
        spearman_rho(r1, r2, nan_policy="omit") for (_, r1), (_, r2) in zip(rf1.items(), rf2.items())
    ])


def average_spearman(rf1: pd.DataFrame, rf2: pd.DataFrame) -> np.array:
    return np.nanmean(list_spearman(rf1, rf2))


class SpearmanScorer(_PredictScorer):
    """
    Scorer object that can be used to evaluate model performance. It calculates
    the score by comparing the true and predicted rankings using the Spearman's
    correlation coefficient. In case a transformer modified the target in the
    training process, the object is submitted and can be used to revert the
    transformation.

    Parameters
    ----------
    transformer: (Type[BaseEstimator] & Type[TransformerMixin])
        A transformer that was previously applied to the target.
    """
    def __init__(
            self,
            factors=None,
            transformer=None
    ):
        super().__init__(None, 1, None)
        if factors is None:
            factors = ['dataset', 'model', 'tuning', 'scoring']
        self.factors = factors
        self.transformer = transformer

    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a `predict`
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        if len(y_true.shape) == 1:
            y_true = pd.Series(y_true, index=y_true.index, name=y_true.name)
            y_pred = pd.Series(method_caller(estimator, "predict", X), index=y_true.index, name=y_true.name)
        else:
            y_true = pd.DataFrame(y_true, index=y_true.index, columns=y_true.columns)
            y_pred = pd.DataFrame(method_caller(estimator, "predict", X), index=y_true.index, columns=y_true.columns)

        if self.transformer is not None:
            _, y_true = self.transformer.inverse_transform(X, y_true)
            _, y_pred = self.transformer.inverse_transform(X, y_pred)

        y_true = pd.Series(
            y_true,
            index=y_true.index,
        )
        y_pred = pd.Series(
            y_pred,
            index=y_true.index,
        )
        y_true.name = 'target'
        y_pred.name = 'target_pred'

        df = pd.concat([X, y_true, y_pred], axis=1)
        y_true_rankings = get_rankings(
            df=df,
            factors=self.factors,
            new_index='encoder',
            target=y_true.name
        )
        y_pred_rankings = get_rankings(
            df=df,
            factors=self.factors,
            new_index='encoder',
            target=y_pred.name
        )

        return average_spearman(y_true_rankings, y_pred_rankings)
