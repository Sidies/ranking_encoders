import pandas as pd
import numpy as np
import enum
import src.features.pairwise_utils as pu
from category_encoders.one_hot import OneHotEncoder
from joblib import dump, load
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, make_scorer, mean_absolute_error, \
    mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV

from src import configuration as config
from src.pipeline.evaluation import evaluation_utils as er
from src.pipeline.evaluation.custom_grid_search import custom_grid_search
from src.features.pairwise_utils import prepare_data


class EvaluationType(enum.Enum):
    BASIC = "basic"
    CROSS_VALIDATION = "cross_validation"
    GRID_SEARCH = "grid_search"
    BAYES_SEARCH = "bayes_search"


class ModelPipeline:
    """
    Class that handles the data pipeline
    """
    _run_count: int = 0

    def __init__(
            self,
            df: pd.DataFrame,
            target: str = "cv_score",
            steps=[],
            verbose_level=0,
            evaluation: EvaluationType = EvaluationType.BASIC,
            X_test: pd.DataFrame = None,
            split_factors=["dataset", "model", "tuning", "scoring"],
            param_grid=[],
            target_transformer=None,
            original_target=None,
            scorer=None,
            n_folds=5,
            bayes_n_iter=200,
            bayes_n_points=4,
            bayes_cv=4,
            bayes_n_jobs=-1,
            workers=1,
            as_pairwise=False
    ):
        """
        Initialize the data pipeline

        Parameters
        ----------
        df: pd.DataFrame
            The dataframe to use for the pipeline
        target: str
            The target column name
        steps: list of tuples
            The tuples should be of the form (name, transformer)
        verbose_level: int
            The verbosity level
        evaluation: EvaluationType
            The type of evaluation to perform.
            Can be one of "basic", "cross_validation", "grid_search"
        split_factors: list of str
            The factors to use for splitting the data into training and validation data
        param_grid: dict
            The parameter grid to use for grid search. Only used if evaluation is "grid_search"
        n_folds: int
            The number of folds to use for cross validation. Only used if evaluation is "cross_validation"
        workers: int
            The number of workers to use for parallel processing
        as_pairwise: bool
            Whether the target is pairwise or not.
        """
        self._pipeline = Pipeline(steps=steps)
        self._df = df
        self._target = target
        self._evaluation = evaluation
        self._verbose_level = verbose_level
        self.X_test = X_test
        self._split_factors = split_factors
        self._target_transformer = target_transformer
        self._original_target = original_target
        self._scorer = scorer
        self._param_grid = param_grid
        self._n_folds = n_folds
        self._bayes_n_iter = bayes_n_iter
        self._bayes_n_points = bayes_n_points
        self._bayes_cv = bayes_cv
        self._bayes_n_jobs = bayes_n_jobs
        self._workers = workers
        self._as_pairwise = as_pairwise

    # start the pipeline
    def run(self):

        print(f"Starting pipeline using method: {self._evaluation}") if self._verbose_level > 0 else None

        self._validation_performance_scores = {}

        performance_metrics = {}
        is_regression = False
        if is_regressor(self._pipeline.named_steps['estimator']):
            is_regression = True
        elif is_classifier(self._pipeline.named_steps['estimator']):
            performance_metrics = {
                'mcc': make_scorer(matthews_corrcoef),
                'accuracy': 'accuracy',
                'f1-score': 'f1_macro'
            }

        if self._evaluation == EvaluationType.BASIC:
            # split the data into training and validation data

            if self._split_factors == []:
                X_train, y_train = self._split_target(self._df, self._target)
                X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                
            else:
                if self._as_pairwise:
                    df_train, df_test = train_test_split(self._df, test_size=0.2, random_state=42)
                    X_train, X_test, y_train = self._prepare_pairwise_data(df_train, df_test, self._target, self._split_factors)
                else:
                    data = self._df
                    X_train, X_test, y_train, y_test = er.custom_train_test_split(
                        data,
                        self._split_factors,
                        self._target,
                        train_size=0.75,
                        random_state=42
                    )
                    
            if self._as_pairwise:
                    y_pred = pd.DataFrame(self._pipeline.fit(X_train, y_train).predict(X_test), columns=y_train.columns, index=X_test.index)
                    df_pred = pd.merge(df_test,
                                    pu.join_pairwise2rankings(X_test, y_pred, self._split_factors),
                                    on=self._split_factors + ["encoder"], how="inner")
                    # use average spearman metric
                    new_index = "encoder"
                    rankings_test = er.get_rankings(df_pred, factors=self._split_factors, new_index=new_index, target=self._target)
                    rankings_pred = er.get_rankings(df_pred, factors=self._split_factors, new_index=new_index, target=self._target + "_pred")
                    self._validation_performance_scores['validation_average_spearman'] = er.average_spearman(
                    rankings_test,
                    rankings_pred
                )

            elif not is_regression:
                self._pipeline.fit(X_train, y_train)
                y_pred = self._pipeline.predict(X_test)
                self._validation_performance_scores['validation_accuracy'] = [accuracy_score(y_test, y_pred)]
                self._validation_performance_scores['validation_f1-score'] = [f1_score(y_test, y_pred, average='macro')]
                self._validation_performance_scores['validation_mcc'] = [matthews_corrcoef(y_test, y_pred)]

            elif is_regression:
                y_pred = self._pipeline.fit(X_train, y_train).predict(X_test)
                self._validation_performance_scores['validation_rmse'] = [
                    mean_squared_error(y_test, y_pred, squared=False)
                ]
                self._validation_performance_scores['validation_mae'] = [mean_absolute_error(y_test, y_pred)]
                self._validation_performance_scores['validation_r2'] = [r2_score(y_test, y_pred)]

                # use average spearman metric
                new_index = "encoder"
                y_pred = pd.Series(
                    self._pipeline.fit(X_train, y_train).predict(X_test),
                    index=y_test.index,
                    name="cv_score_pred"
                )
                df_pred = pd.concat([X_test, y_test, y_pred], axis=1)
                # ---- convert to rankings and evaluate
                rankings_test = er.get_rankings(
                    df_pred,
                    factors=self._split_factors,
                    new_index=new_index,
                    target="cv_score"
                )
                rankings_pred = er.get_rankings(
                    df_pred,
                    factors=self._split_factors,
                    new_index=new_index,
                    target="cv_score_pred"
                )
                self._validation_performance_scores['validation_average_spearman'] = er.average_spearman(
                    rankings_test,
                    rankings_pred
                )

        elif self._evaluation == EvaluationType.CROSS_VALIDATION:
            if self._y_train is None:
                    X_train, y_train = self._split_target(self._df, self._target)
            else:
                y_train = self._y_train
                X_train = self._df
            self._pipeline.fit(X_train, y_train)
            self._validation_performance_scores = er.custom_cross_validation(
                self._pipeline,
                self._df,
                self._split_factors,
                self._target,
                self._target_transformer,
                self._scorer,
                cv=self._n_folds,
                verbose=self._verbose_level
            )

        elif self._evaluation == EvaluationType.GRID_SEARCH:
            if self._y_train is None:
                    X_train, y_train = self._split_target(self._df, self._target)
            else:
                y_train = self._y_train
                X_train = self._df
            self._validation_performance_scores = self._do_grid_search(X_train, y_train, param_grid=self._param_grid)

        elif self._evaluation == EvaluationType.BAYES_SEARCH:
            if self._y_train is None:
                    X_train, y_train = self._split_target(self._df, self._target)
            else:
                y_train = self._y_train
                X_train = self._df
            self._validation_performance_scores = self._do_bayes_search(
                X_train,
                y_train,
                param_grid=self._param_grid,
                scoring=self._scorer,
                n_iter=self._bayes_n_iter,
                n_points=self._bayes_n_points,
                cv=self._bayes_cv,
                n_jobs=self._bayes_n_jobs
            )

        else:
            raise Exception(f"Unknown evaluation type: {self._evaluation}")

        print("Finished running the pipeline") if self._verbose_level > 0 else None
        self._run_count += 1

        # print out the evaluation metrics
        if self._verbose_level > 0 and self._validation_performance_scores != {}:
            print("Evaluation metrics:")
            # print the evaluation metrics
            if self._evaluation == EvaluationType.CROSS_VALIDATION:
                for metric, value in self._validation_performance_scores.items():
                    if isinstance(value, float) or isinstance(value, int):
                        output = metric + ': ' + (str(round(value, 4)))
                    else:
                        output = metric + ': ' + str(value)
                    print("    " + output)

                # print average of all folds
                print("    average of all folds: "
                      + str(round(np.mean(list(self._validation_performance_scores.values())), 4))
                      + ' [std=' + str(round((np.std(list(self._validation_performance_scores.values()))), 4)) + ']')
            elif self._evaluation == EvaluationType.GRID_SEARCH:
                for metric, value in self._validation_performance_scores.items():
                    if isinstance(value, float) or isinstance(value, int):
                        output = metric + ': ' + (str(round(value, 4)))
                    else:
                        output = metric + ': ' + str(value)
                    print("    " + output)
            else:
                for metric, values in {**self._validation_performance_scores}.items():
                    output = metric + ': ' + np.array2string(np.mean(values), precision=4) \
                             + ' [std=' + np.array2string(np.std(values), precision=4) + ']'
                    print("    " + output)

        # save the predictions to a file if X_test is not None
        if self.X_test is not None:
            self.save_prediction(self.X_test)

    # get the pipeline
    def get_pipeline(self):
        return self._pipeline

    def predict(self, df):
        """
        Make predictions.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        list
            Predictions made by the model.
        """
        if not self._is_initialized():
            raise Exception('The pipeline is not initialized yet. Please run the pipeline first.')

        return self._pipeline.predict(df)

    def add_new_step(self, transformer, name):
        """
        Adds a new step to the pipeline before the 'estimator' step, if it
        exists. If the name is already present, the corresponding
        transformer will be replaced.

        Args:
            transformer (transformer): the transformer to be added
            name (string): name of the step
        """
        # initialize new pipeline if empty
        if len(self._pipeline.steps) == 0:
            self._create_pipeline([(name, transformer)])
            return

        # insert step before estimator or replace if step already exists
        insertion_index = len(self._pipeline.steps)  # insert at end of list by default
        pip_steps = self._pipeline.steps
        for i, (step_name, _) in enumerate(self._pipeline.steps):
            if step_name == 'estimator':
                insertion_index = i
                break
            if step_name == name:
                pip_steps[i] = (name, transformer)  # replace the step
                self._create_pipeline(steps=pip_steps)
                return  # no need to insert if step already exists
        pip_steps.insert(insertion_index, (name, transformer))
        self._create_pipeline(steps=pip_steps)

    def add_new_step_at_position(self, transformer, name, position: int):
        """
        Adds a new step to the pipeline at a specific position. If the name is already present, the corresponding
        transformer will be replaced. If the position is out of bounds, the step will be appended to the end of the
        pipeline.

        Args:
            transformer (transformer): the transformer to be added
            name (string): name of the step
            position (int): position in the pipeline
        """
        # initialize steps if empty
        if len(self._pipeline.steps) == 0:
            self._create_pipeline(steps=[(name, transformer)])
            return

        # replace if step already exists
        pip_steps = self._pipeline.steps
        for i, (step_name, _) in enumerate(pip_steps):
            if step_name == name:
                pip_steps[i] = (name, transformer)
                self._create_pipeline(steps=pip_steps)
                return

        # if name not found, insert or append step
        if position >= len(pip_steps):
            pip_steps.append((name, transformer))
        else:
            pip_steps.insert(position, (name, transformer))
        self._create_pipeline(steps=pip_steps)

    def remove_step(self, name: str):
        """Removes a specific step from the pipeline

        Args:
            name (string): the name of the step to remove
        """
        pip_steps = self._pipeline.steps
        for i, (step_name, _) in enumerate(pip_steps):
            if step_name == name:
                pip_steps.pop(i)
                break
        self._create_pipeline(steps=pip_steps)

    def clear_steps(self):
        """Removes all steps from the pipeline except the 'estimator' step
        """
        pip_steps = self._pipeline.steps
        for i, (step_name, _) in enumerate(pip_steps):
            if step_name == 'estimator':
                pip_steps = pip_steps[i:]
                break
        self._create_pipeline(steps=pip_steps)

    def change_estimator(self, new_estimator):
        """Changes the current estimator of the pipeline to a different one.
        If no estimator is defined a new one will be added.

        Args:
            new_estimator (model): the new model that should be applied
        """
        # Check if there is an 'estimator' step already defined
        pip_steps = self._pipeline.steps
        for i, step in enumerate(pip_steps):
            if step[0] == 'estimator':
                # Replace the estimator
                pip_steps[i] = ('estimator', new_estimator)
                break
        # the else is executed if the for loop is not broken
        else:
            # If no 'estimator' step was found, add one
            pip_steps.append(('estimator', new_estimator))
        self._create_pipeline(steps=pip_steps)

    def save_pipeline(self, path):
        """
        Save the pipeline to a file.

        Parameters
        ----------
        path: string
            The path the pipeline should be stored to
        """

        if not self._is_initialized():
            raise Exception('The pipeline is not initialized yet. Please run the pipeline first.')

        dump(self._pipeline, path)

    def load_pipeline(self, path):
        """
        Load the pipeline from a file

        Parameters
        ----------
        path: string
            The path the pipeline should be loaded from
        """
        self._pipeline = load(path)

    def save_prediction(self, data: pd.DataFrame):
        """
        Save the predictions to a file at location data/processed.

        Args:
            data : The data to make predictions on
        """

        # check if df contains the target column(s) and remove if necessary
        if isinstance(self._target, list):
            target = set(self._target)
        elif isinstance(self._target, pd.Index):
            target = set(list(self._target))
        else:
            target = {self._target}
        if self._original_target is not None:
            if isinstance(self._original_target, list):
                target = target = target | set(self._original_target)
            elif isinstance(self._original_target, pd.Index):
                target = target = target | set(list(self._original_target))
            else:
                target = target = target | {self._original_target}
        data = data.drop(list(set(data.columns).intersection(target)), axis=1)

        # check if the pipeline is initialized
        if not self._is_initialized():
            raise Exception('The pipeline is not initialized yet. Please run the pipeline first.')

        # get the predictions
        predictions = self._pipeline.predict(data)

        # save the predictions
        path = config.DATA_DIR / 'processed/regression_tyrell_prediction.csv'
        print("Saving predictions to {}".format(path)) if self._verbose_level > 0 else None
        pd.DataFrame(predictions).to_csv(path, index=False, header=False)

    def _create_pipeline(self, steps):
        self._run_count = 0
        self._pipeline = Pipeline(steps=steps)

    def _is_initialized(self):
        """
        Check if the pipeline is initialized
        """
        is_initialized = self._pipeline is not None

        if self._run_count == 0:
            is_initialized = False

        return is_initialized

    def _do_grid_search(self, X_train: pd.DataFrame, y_train, param_grid, cv=5, n_jobs=-1, scoring=None):
        """
        Perform grid search on the pipeline

        Parameters
        ----------
        param_grid: dict
            The parameter grid
        scoring: str
            The scoring function
        cv: int
            The number of folds for cross validation
        n_jobs: int
            The number of jobs to run in parallel
        """
        print("Performing grid search") if self._verbose_level > 0 else None
        validation_performance_scores = {}

        results, best_params, max_score = custom_grid_search(
            self._pipeline,
            self._df,
            self._param_grid,
            self._split_factors,
            self._target,
            target_transformer=self._target_transformer,
            scorer=self._scorer,
            cv=self._n_folds,
            ParallelUnits=self._workers,
            verbose=self._verbose_level
        )

        validation_performance_scores["best_score"] = max_score
        validation_performance_scores["best_params"] = best_params
        if self._verbose_level > 2:
            validation_performance_scores["results_per_parameter"] = results

        return validation_performance_scores

    def _do_bayes_search(
            self,
            X_train: pd.DataFrame,
            y_train, param_grid,
            scoring=er.RegressionSpearmanScorer,
            cv=4,
            n_iter=200,
            n_points=4,
            n_jobs=-1,
            random_state=None
    ):
        """
        Perform bayes search on the pipeline

        Parameters
        ----------
        param_grid: dict
            The parameter grid
        scoring: sklearn.metrics._scorer._BaseScorer
            The scoring object
        cv: int
            The number of folds for cross validation
        n_iter: int
            The total number of fits
        n_points: int
            The number of fits performed per round
        n_jobs: int
            The number of jobs to run in parallel
        """
        print("Performing bayes search") if self._verbose_level > 0 else None
        validation_performance_scores = {}

        indices = er.custom_cross_validated_indices(
            df=self._df,
            factors=self._split_factors,
            target=self._target,
            n_splits=cv,
            shuffle=True,
            random_state=random_state
        )

        search = BayesSearchCV(
            estimator=self._pipeline,
            search_spaces=param_grid,
            scoring=scoring,
            cv=indices,
            n_iter=n_iter,
            n_points=n_points,
            n_jobs=n_jobs,
            refit=False,
            verbose=self._verbose_level,
            random_state=random_state
        )
        search.fit(X_train, y_train)

        print('best score: ' + str(search.best_score_))
        print('best params:')
        for key, value in search.best_params_.items():
            print('    ' + str(key) + ': ' + str(value))

        print('Training pipeline with best parameters...')
        self._pipeline.set_params(**search.best_params_)
        self._pipeline.fit(X_train, y_train)

        print('Evaluating pipeline with best parameters...')
        validation_performance_scores = er.custom_cross_validation(
            self._pipeline,
            self._df,
            self._split_factors,
            self._target,
            target_transformer=self._target_transformer,
            scorer=self._scorer,
            cv=self._n_folds,
            verbose=self._verbose_level
        )
        validation_performance_scores['average_spearman (5-fold)'] = list(validation_performance_scores.values())

        return validation_performance_scores

    def _split_target(self, df, target):
        """
        Split the target from the dataframe

        Parameters
        ----------
        df: pandas.DataFrame
            The dataframe to split the target from
        target: str
            The name of the target column
        """
        X_train = df.drop(target, axis=1)
        y_train = df[target]

        return X_train, y_train

    def _prepare_pairwise_data(self, df_train, df_test, target, split_factors):
        X_train, X_test, y_train = prepare_data(df_train, df_test, target=target, factors=split_factors)
        return X_train, X_test, y_train