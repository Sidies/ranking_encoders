"""
The pairwise prediction task is made of two parts.

--- First task
First, transform df_train and df_test in a suitable format.
    X_train and X_test have columns ["dataset", "model", "tuning", "scoring"]
    y_train has a column for each ordered pair of distinct encoders (992 columns)
Second, train a multioutput model on X_train, y_train and get prediction y_pred from X_test.
    Transform y_pred into rankings and join with df_test
    Evaluate with average_spearman
"""

import numpy as np
import pandas as pd

from category_encoders import OneHotEncoder
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from src import configuration as config

import src.features.encoder_utils as eu
import src.pipeline.evaluation.evaluation_utils as er
import src.features.pairwise_utils as pu
from sklearn.model_selection import train_test_split

DATA_DIR = config.DATA_RAW_DIR

factors = ["dataset", "model", "tuning", "scoring"]
new_index = "encoder"

df_train = config.load_dataset(DATA_DIR / "dataset_rank_train.csv")
#df_test = ld.load_dataset(DATA_DIR / "dataset_rank_test.csv")  # as usual, replace it with your own validation set
df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42)
X_train = df_train[factors + ["encoder"]].groupby(factors).agg(lambda x: np.nan).reset_index()[factors]
X_test = df_test[factors + ["encoder"]].groupby(factors).agg(lambda x: np.nan).reset_index()[factors]

# join to ensure X_train and y_train's indices are ordered the same
y_train = pd.merge(X_train,
                   pu.get_pairwise_target(df_train, features=factors, target="rank", column_to_compare="encoder"),
                   on=factors, how="left").drop(factors, axis=1).fillna(0)

print(f"y_train shape: {y_train.shape}")

dummy_pipe = Pipeline([("encoder", eu.NoY(OneHotEncoder())), ("model", DecisionTreeClassifier())])
y_pred = pd.DataFrame(dummy_pipe.fit(X_train, y_train).predict(X_test), columns=y_train.columns, index=X_test.index)
df_pred = pd.merge(df_test,
                   pu.join_pairwise2rankings(X_test, y_pred, factors),
                   on=factors + ["encoder"], how="inner")

print(df_pred.head(50))
print(df_pred["rank_pred"].value_counts())

rankings_test = er.get_rankings(df_pred, factors=factors, new_index=new_index, target="rank")
rankings_pred = er.get_rankings(df_pred, factors=factors, new_index=new_index, target="rank_pred")
print(er.average_spearman(rankings_test, rankings_pred))


