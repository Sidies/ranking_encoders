import numpy as np
import pandas as pd
import warnings

from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics._scorer import _PredictScorer
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from typing import Iterable, List

from src.features.pairwise_utils import prepare_data, join_pairwise2rankings
from src.pipeline.evaluation.evaluation_utils import average_spearman, get_rankings, custom_train_test_split

def custom_cross_validation(
        pipeline: Pipeline,
        df,
        factors,
        target,
        target_transformer=None,
        scorer=None,
        train_size=0.75,
        shuffle=True,
        cv=5,
        verbose=1,
        as_pairwise=False
):
    """
    Like cross_validation, but keeps encoders together: the split is on 'factors'.
    """
    # check if tqdm should be disabled
    disable_tqdm = (verbose == 0)
    validation_performance_scores = {}
    for i in tqdm(range(cv), disable=disable_tqdm):         
        X_train, X_test, y_train, y_test = custom_train_test_split(
            df,
            factors,
            target,
            train_size=train_size,
            shuffle=shuffle,
            random_state=i
        )

        if scorer is not None:
            pipeline.fit(X_train, y_train)
            validation_performance_scores[
                'validation_average_spearman_fold_' + str(i)
            ] = scorer(pipeline, X_test, y_test)
        else:
            if target_transformer is not None:
                _, y_train = target_transformer.fit_transform(X_train, y_train)

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            if target_transformer is not None:
                _, y_pred = target_transformer.inverse_transform(X_test, y_pred)

            y_pred = pd.Series(y_pred, index=y_test.index, name=target + "_pred")

            # ---- convert to rankings and evaluate
            if target == "cv_score":
                new_index = "encoder"
                df_pred = pd.concat([X_test, y_test, y_pred], axis=1)
                rankings_test = get_rankings(df_pred, factors=factors, new_index=new_index, target=target)
                rankings_pred = get_rankings(df_pred, factors=factors, new_index=new_index, target=target + "_pred")
            else:
                rankings_test = pd.DataFrame(y_test)
                rankings_pred = pd.DataFrame(y_pred)

            validation_performance_scores['validation_average_spearman_fold_' + str(i)] = average_spearman(
                rankings_test,
                rankings_pred
            )
              
    return validation_performance_scores


def pairwise_custom_cross_validation(
        pipeline: Pipeline,
        df,
        factors,
        target,
        train_size=0.75,
        shuffle=True,
        cv=5,
        verbose=1,
        as_pairwise=False
):
    """
    Like cross_validation, but with pairwise target.
    """
    # check if tqdm should be disabled
    disable_tqdm = (verbose == 0)
    validation_performance_scores = {}
    for i in tqdm(range(cv), disable=disable_tqdm):         
        df_train, df_test = train_test_split(df, train_size=0.2, random_state=i)
        X_train, X_test, y_train = prepare_data(df_train, df_test, target=target, factors=factors)

        y_pred = pd.DataFrame(pipeline.fit(X_train, y_train).predict(X_test), columns=y_train.columns, index=X_test.index)
        
        df_pred = pd.merge(df_test,
                            join_pairwise2rankings(X_test, y_pred, factors),
                            on=factors + ["encoder"], how="inner")
        # use average spearman metric
        new_index = "encoder"
        rankings_test = get_rankings(df_pred, factors=factors, new_index=new_index, target=target)
        rankings_pred = get_rankings(df_pred, factors=factors, new_index=new_index, target=target + "_pred")
        
        validation_performance_scores['validation_average_spearman_fold_' + str(i)] = average_spearman(
            rankings_test,
            rankings_pred
        )
              
    return validation_performance_scores