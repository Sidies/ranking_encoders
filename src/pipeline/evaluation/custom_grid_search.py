import os
import sys
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from src.pipeline.evaluation.evaluate_regression import get_rankings, average_spearman, custom_train_test_split
from sklearn.pipeline import Pipeline


def custom_grid_search(pipeline: Pipeline, df, GridSearchParams, split_factors, target:str, cv=5, train_size=0.75, ParallelUnits=1):
    """
    Perform a grid search on the given model using the specified parameters,
    returning the results, the best parameters, and the best score.

    Parameters
    ----------
    Model : sklearn model
        The model to perform the grid search on
    X : array-like, shape (n_samples, n_features)
        The training input samples
    y : array-like, shape (n_samples,)
        The target values
    GridSearchParams : dict
        The parameters to search over, with keys as the parameter names and
        values as a list of possible values
    cv : int, optional (default=5)
        The number of cross-validation splits to perform
    ParallelUnits : int, optional (default=1)
        The number of parallel workers to use. If -1, use all available CPU cores
    error_metrics : str, optional (default="mse")
        The error metric to use when determining the best parameters. Can be
        one of "mae", "mse", "rmse", or "r2"
    random_state : int, optional (default=None)
        Seed for random number generator

    Returns
    -------
    results : list
        A list of tuples, each containing a dictionary of parameters and the
        corresponding average error
    best_params : dict
        The best parameters found
    error : float
        The best score found

    """

    # Train the model using the specified parameters and calculate the error
    def process_param_set(params, X_train, X_test, y_train, y_test, pipeline:Pipeline, split_factors, target):

        """
        Train the model using the specified parameters and calculate the error.

        Parameters
        ----------
        args : tuple
            A tuple containing the parameters, input data, target data, error metric,
            train indices, and test indices
        Model : custom model
            The model to train
        error_metrics_dict : dict
            A dictionary mapping error metric names to their corresponding functions

        Returns
        -------
        tuple
            A tuple of the parameters and the calculated error

        """
        
        # set the parameters
        pipeline.set_params(**params)

        # Train the pipeline     
        y_pred = pd.Series(pipeline.fit(X_train, y_train).predict(X_test), index=y_test.index, name="cv_score_pred")
        
        # calculate average spearman correlation
        new_index = "encoder"
        df_pred = pd.concat([X_test, y_test, y_pred], axis=1)
        # ---- convert to rankings and evaluate
        rankings_test = get_rankings(df_pred, factors=split_factors, new_index=new_index, target=target)
        rankings_pred = get_rankings(df_pred, factors=split_factors, new_index=new_index, target=target + "_pred")
        
        error = average_spearman(rankings_test, rankings_pred)

        return params, error
    

    # Initialize variables to store the results
    Results = []
    Keys = GridSearchParams.keys()
    MaxScore = -sys.maxsize
    BestParams = {}
    ParamSets = [dict(zip(Keys, values)) for values in itertools.product(*GridSearchParams.values())]

    # Check the number of parallel units
    if ParallelUnits == -1:
        ParallelUnits = os.cpu_count()
    elif ParallelUnits > os.cpu_count():
        raise RuntimeError(f"Runtime Error: {os.cpu_count()}")

    # Start the thread pool executor
    try:
        with ThreadPoolExecutor(max_workers=ParallelUnits) as executor:
            for params in tqdm(ParamSets):
                total_error = 0
                futures = []

                for i in tqdm(range(cv)):
                    X_train, X_test, y_train, y_test = custom_train_test_split(df, split_factors, target, train_size, random_state=i)
                    future = executor.submit(process_param_set,
                                             params, X_train, X_test, y_train, y_test, pipeline, split_factors, target)
                    futures.append(future)

                for future in futures:
                    _, error = future.result()
                    total_error += error

                avg_error = total_error / cv
                Results.append((params, avg_error))

                # we try to get the highest score
                if avg_error > MaxScore:
                    MaxScore = avg_error
                    BestParams = params
                

    # Handle keyboard interrupt to stop the execution
    except KeyboardInterrupt:
        print("CTRL+C pressed. Stopping the execution...")
        executor.shutdown(wait=False)

    # Return the results, best parameters and error
    return Results, BestParams, MaxScore
