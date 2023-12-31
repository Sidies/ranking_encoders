import os
import sys
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from src.pipeline.evaluation.evaluation_utils import get_rankings, average_spearman, custom_train_test_split
from src.features.pairwise_utils import prepare_data, join_pairwise2rankings
from src.features.encoder_utils import NoY
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def custom_grid_search(
    pipeline: Pipeline,
    df, GridSearchParams,
    split_factors,
    target:str,
    target_transformer=None,
    scorer=None,
    cv=5,
    train_size=0.75,
    ParallelUnits=1,
    verbose=1):
    """
    Perform a grid search on the given model using the specified parameters,
    returning the results, the best parameters, and the best score.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The pipeline to perform the grid search on
    df : pandas.DataFrame
        The input data as a pandas DataFrame
    GridSearchParams : dict
        A dictionary of hyperparameters to search over, where the keys are the
        names of the hyperparameters and the values are lists of possible values
    split_factors : list of str
        The column names to use for splitting the data into training and test sets
    target : str
        The name of the target variable column
    cv : int, optional (default=5)
        The number of cross-validation splits to perform
    train_size : float, optional (default=0.75)
        The proportion of the data to use for training
    ParallelUnits : int, optional (default=1)
        The number of parallel workers to use. If -1, use all available CPU cores
    verbose : int, optional (default=1)
        The verbosity level of the output. Set to 0 for no output, 1 for some output,
        and 2 for detailed output.

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
    def process_param_set(
        params,
        X_train,
        X_test,
        y_train,
        y_test,
        pipeline:Pipeline,
        split_factors,
        target,
        target_transformer,
        scorer
    ):
        # set the parameters
        pipeline.set_params(**params)

        if scorer is not None:
            pipeline.fit(X_train, y_train)
            error = scorer(pipeline, X_test, y_test)
        else:
            if target_transformer is not None:
                _, y_train = target_transformer.fit_transform(X_train, y_train)

            # Train the pipeline
            pipeline.fit(X_train, y_train)
            y_pred = pd.Series(pipeline.predict(X_test), index=y_test.index, name=target + "_pred")

            if target_transformer is not None:
                _, y_pred = target_transformer.inverse_transform(X_test, y_pred)

            new_index = "encoder"
            df_pred = pd.concat([X_test, y_test, y_pred], axis=1)
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
    
    # check if tqdm should be disabled
    disable_tqdm = (verbose == 0)
        

    # Check the number of parallel units
    if ParallelUnits == -1:
        ParallelUnits = os.cpu_count()
    elif ParallelUnits > os.cpu_count():
        raise RuntimeError(f"Runtime Error: {os.cpu_count()}")

    # Start the thread pool executor
    try:
        with ThreadPoolExecutor(max_workers=ParallelUnits) as executor:
            for params in tqdm(ParamSets, disable=disable_tqdm):
                try:
                    total_error = 0
                    futures = []

                    for i in tqdm(range(cv), disable=disable_tqdm, leave=False):
                        X_train, X_test, y_train, y_test = custom_train_test_split(df, split_factors, target, train_size, random_state=i)
                        future = executor.submit(
                            process_param_set,
                            params,
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            pipeline,
                            split_factors,
                            target,
                            target_transformer,
                            scorer
                        )
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
                except Exception as e:
                    print(e)
                    
    # Handle keyboard interrupt to stop the execution
    except KeyboardInterrupt:
        print("CTRL+C pressed. Stopping the execution...")
        executor.shutdown(wait=False)
    except Exception as e:
        print("An error occurred.")
        print(e)
        

    # Return the results, best parameters and error
    return Results, BestParams, MaxScore


def pairwise_custom_grid_search(
    pipeline: Pipeline,
    df, GridSearchParams,
    split_factors,
    target:str,
    target_transformer=None,
    scorer=None,
    cv=5,
    train_size=0.75,
    ParallelUnits=1,
    verbose=1):
    """
    Perform a grid search on the given model using the specified parameters,
    returning the results, the best parameters, and the best score.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The pipeline to perform the grid search on
    df : pandas.DataFrame
        The input data as a pandas DataFrame
    GridSearchParams : dict
        A dictionary of hyperparameters to search over, where the keys are the
        names of the hyperparameters and the values are lists of possible values
    split_factors : list of str
        The column names to use for splitting the data into training and test sets
    target : str
        The name of the target variable column
    cv : int, optional (default=5)
        The number of cross-validation splits to perform
    train_size : float, optional (default=0.75)
        The proportion of the data to use for training
    ParallelUnits : int, optional (default=1)
        The number of parallel workers to use. If -1, use all available CPU cores
    verbose : int, optional (default=1)
        The verbosity level of the output. Set to 0 for no output, 1 for some output,
        and 2 for detailed output.

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
    def process_param_set(
        params,
        X_train,
        X_test,
        y_train,
        pipeline:Pipeline,
        split_factors,
        target,
    ):
        # set the parameters
        pipeline.set_params(**params)

        if target_transformer is not None:
            _, y_train = target_transformer.fit_transform(X_train, y_train)

        # Train the pipeline
        y_pred = pd.DataFrame(pipeline.fit(X_train, y_train).predict(X_test), columns=y_train.columns, index=X_test.index)
        
        df_pred = pd.merge(df_test,
                            join_pairwise2rankings(X_test, y_pred, split_factors),
                            on=split_factors + ["encoder"], how="inner")
        # use average spearman metric
        new_index = "encoder"
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
    
    # check if tqdm should be disabled
    disable_tqdm = (verbose == 0)
        

    # Check the number of parallel units
    if ParallelUnits == -1:
        ParallelUnits = os.cpu_count()
    elif ParallelUnits > os.cpu_count():
        raise RuntimeError(f"Runtime Error: {os.cpu_count()}")

    # Start the thread pool executor
    try:
        with ThreadPoolExecutor(max_workers=ParallelUnits) as executor:
            for params in tqdm(ParamSets, disable=disable_tqdm):
                try:
                    total_error = 0
                    futures = []

                    for i in tqdm(range(cv), disable=disable_tqdm, leave=False):
                        X_train, X_test, y_train, y_test = custom_train_test_split(df, factors=split_factors, target=target, train_size=0.8, random_state=42)

                        df_train = X_train.copy()
                        df_train[target] = y_train
                        df_test = X_test.copy()
                        df_test[target] = y_test

                        X_train, X_test, y_train = prepare_data(df_train, df_test, target=target, factors=split_factors)
                        
                        future = executor.submit(
                            process_param_set,
                            params,
                            X_train,
                            X_test,
                            y_train,   
                            pipeline,
                            split_factors,
                            target,                      
                        )
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
                except Exception as e:
                    print(e)
                    
    # Handle keyboard interrupt to stop the execution
    except KeyboardInterrupt:
        print("CTRL+C pressed. Stopping the execution...")
        executor.shutdown(wait=False)
    except Exception as e:
        print("An error occurred.")
        print(e)
        

    # Return the results, best parameters and error
    return Results, BestParams, MaxScore
