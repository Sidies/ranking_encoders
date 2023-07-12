import pandas as pd
import numpy as np

from functools import reduce
from itertools import product

import src.pipeline.evaluation.evaluation_utils as er


def get_pairwise_target(df, features, target, column_to_compare):
    """
    Given a dataframe, a list of features, a target column and a column to compare, this function returns a new dataframe where each row represents a pair of encoders and the target column indicates which encoder is better.

    Args:
        df (pd.DataFrame): The input dataframe.
        features (list): A list of feature column names.
        target (str): The target column name.
        column_to_compare (str): The column to compare the encoders.

    Returns:
        pd.DataFrame: A new dataframe where each row represents a pair of encoders and the target column indicates which
        encoder is better.
    """
    dfs = []
    # Iterate over all unique pairs of encoders
    for e1, e2 in product(df[column_to_compare].unique(), repeat=2):
        # Skip if e1 >= e2 to avoid duplicates
        if e1 >= e2:
            continue
        # Merge the rows of the two encoders on the feature columns
        tmp = pd.merge(df.loc[df[column_to_compare] == e1], df.loc[df[column_to_compare] == e2],
                       on=features, how="inner", suffixes=("_1", "_2"))
        # Create a column indicating which encoder is better
        tmp[(e1, e2)] = (tmp[f"{target}_1"] < tmp[f"{target}_2"]).astype(int)
        tmp[(e2, e1)] = (tmp[f"{target}_2"] < tmp[f"{target}_1"]).astype(int)
        # Append the new dataframe to the list of dataframes
        dfs.append(tmp[features + [(e1, e2), (e2, e1)]])
    # Merge all dataframes in the list into a single dataframe
    return reduce(lambda d1, d2: pd.merge(d1, d2, on=features, how="outer"), dfs)


def join_pairwise2rankings(X: pd.DataFrame, y: pd.DataFrame, factors):
    """
    Fixed ["dataset", "model", "tuning", "scoring"], the rank of an encoder is the _opposite_ of the number
    of encoders that it beats; the more other encodes it beats, the better it is.
    """

    if not y.index.equals(X.index):
        raise ValueError("y and X must have the same index.")

    tmp = pd.concat([X, y], axis=1).melt(id_vars=factors, var_name="encoder_pair",
                                         value_name="is_better_than")
    tmp = pd.concat([tmp.drop("encoder_pair", axis=1),
                     pd.DataFrame(tmp["encoder_pair"].to_list(), columns=["encoder", "encoder_2"])],
                    axis=1)
    tmp = tmp.groupby(factors + ["encoder"])["is_better_than"].sum().reset_index()
    tmp["rank_pred"] = er.score2ranking(tmp["is_better_than"], ascending=False)

    return tmp.drop("is_better_than", axis=1)


def prepare_data(df_train, df_test, factors, target="rank", column_to_compare="encoder"):
    X_train = df_train[factors + ["encoder"]].groupby(factors).agg(lambda x: np.nan).reset_index()[factors]
    X_test = df_test[factors + ["encoder"]].groupby(factors).agg(lambda x: np.nan).reset_index()[factors]
    # join to ensure X_train and y_train's indices are ordered the same
    y_train = pd.merge(X_train,
                    get_pairwise_target(df_train, features=factors, target=target, column_to_compare=column_to_compare),
                    on=factors, how="left").drop(factors, axis=1).fillna(0)
    
    # convert all to string
    #X_train = X_train.astype(str)
    #X_test = X_test.astype(str)
    #y_train = y_train.astype(str)
    
    return X_train, X_test, y_train

