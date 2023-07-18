import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import src.pipeline.evaluation.evaluation_utils as er
import src.configuration as ld

factors = ["dataset", "model", "tuning", "scoring"]
new_index = "encoder"

df = ld.load_traindata_for_pointwise()
X_factors = df.groupby(factors).agg(lambda a: np.nan).reset_index()[factors]
factors_train, factors_test = train_test_split(
    X_factors,
    stratify=X_factors.dataset,
    train_size=0.75,
    shuffle=True,
    random_state=0
)
df_train = pd.merge(factors_train, df, on=factors)
df_test = pd.merge(factors_test, df, on=factors)

df_train = pd.pivot(df_train, index=factors, columns="encoder", values="rank").reset_index()
df_test = pd.pivot(df_test, index=factors, columns="encoder", values="rank").reset_index()

X_train = df_train[factors]
X_test = df_test[factors]
y_train = df_train.drop(factors, axis=1)
y_test = df_test.drop(factors, axis=1)

# Split df_train
# X, y = df_train[factors], df_train.drop(factors, axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

""" fill in the missing target values with the worst possible rank (NaN = the encoder did not terminate).
A better solution would be to impute missing value y[i, j] with np.max(y_train, axis=1)[i].  
The choice of imputation does affect performance!
"""
y_train = y_train.fillna(np.max(y_train))
# y_train = y_train.fillna(100)
# y_train = y_train.fillna(0)

dummy_pipe = Pipeline([("encoder", OneHotEncoder()), ("model", DecisionTreeRegressor(random_state=43))])
y_pred = pd.DataFrame(dummy_pipe.fit(X_train, y_train).predict(X_test), index=X_test.index)
y_pred.columns = y_train.columns

# Evaluation
df_pred = pd.merge(pd.concat([X_test, y_test], axis=1).melt(id_vars=factors, value_name="rank").dropna(axis=0),
                   pd.concat([X_test, y_pred], axis=1).melt(id_vars=factors, value_name="rank_pred"),
                   on=factors+["encoder"], how="left")

rankings_test = er.get_rankings(df_pred, factors=factors, new_index=new_index, target="rank")
rankings_pred = er.get_rankings(df_pred, factors=factors, new_index=new_index, target="rank_pred")
print(f"Average Spearman: {er.average_spearman(rankings_test, rankings_pred):.03f}")