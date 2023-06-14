import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


class ModelPipeline:
    """
    Class that handles the data pipeline
    """
    _run_count:int = 0
    
    def __init__(self, x_train_df, y_train_df, steps=None, verbose_level=0, evaluation:str="basic", param_grid=[]):
        """
        Initialize the data pipeline

        Parameters
        ----------
        x_train_df: pandas.DataFrame
            The training data
        y_train_df: pandas.DataFrame
            The training labels
        steps: list of tuples 
            The tuples should be of the form (name, transformer) 
        verbose_level: int
            The verbosity level
        evaluation: str
            The type of evaluation to perform.
            Can be one of "basic", "cross_validation", "grid_search"
        param_grid: dict
            The parameter grid to use for grid search. Only used if evaluation is "grid_search"
        """        
        self._pipeline = Pipeline(steps=steps)        
        self._x_train_df = x_train_df
        self._y_train_df = y_train_df
        self._evaluation = evaluation
        self._verbose_level = verbose_level
        self._param_grid = param_grid
        
    def get_pipeline_step_decorator(self):
        """
        Decorator to add a step to the pipeline
        """
        def _step_decorator(transformer):
            def _wrapper(name:str, position:int=None, *args, **kwargs):
                """
                Wrapper function to add a step to the pipeline
                
                Parameters
                ----------
                name: str
                    The name of the step
                position: int
                    The position in the pipeline where the step should be added.
                    If None, the step is added to the end of the pipeline.
                """
                step = (name, transformer(*args, **kwargs))
                if self._pipeline is None:
                    self._pipeline = Pipeline([step])
                elif self._pipeline.steps is None:
                    self._pipeline.steps = [step]
                else:
                    steps = list(self._pipeline.steps)
                    if position is not None and -len(steps) <= position <= len(steps):
                        steps.insert(position, step)
                    else:
                        steps.append(step)
                    steps.append(step)
                    self._pipeline = Pipeline(steps)
                return self._pipeline
            return _wrapper
        return _step_decorator
    
    
    # start the pipeline
    def run(self):
        
        print(f"Starting pipeline using method: {self._evaluation}") if self._verbose_level > 0 else None
        
        validation_performance_scores = {}
        performance_metrics = {
                'mcc': make_scorer(matthews_corrcoef),
                'accuracy': 'accuracy',
                'f1-score': 'f1_macro',
                
            }        
        if self._evaluation == "basic":
            # split the data into training and validation data
            self.x_train_df, self.x_test_df, self.y_train_df, self.y_test_df = train_test_split(self._x_train_df, self._y_train_df, test_size=0.2, random_state=42)
            self._pipeline.fit(self._x_train_df, self._y_train_df)
            y_pred = self._pipeline.predict(self.x_test_df)
            validation_performance_scores['validation_accuracy'] = [accuracy_score(self.y_test_df, y_pred)]
            validation_performance_scores['validation_f1-score'] = [f1_score(self.y_test_df, y_pred, average='macro')]
            validation_performance_scores['validation_mcc'] = [matthews_corrcoef(self.y_test_df, y_pred)]
            
        elif self._evaluation == "cross_validation":
            self._pipeline.fit(self._x_train_df, self._y_train_df)
            validation_performance_scores = cross_validate(
                self._pipeline,
                self._x_train_df,
                self._y_train_df,
                scoring=performance_metrics,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            )
            
        elif self._evaluation == "grid_search":
            validation_performance_scores = self._do_grid_search(param_grid=self._param_grid, scoring=performance_metrics)
            
        else:
            raise Exception(f"Unknown evaluation type: {self._evaluation}")
            
        print("Finished running the pipeline") if self._verbose_level > 0 else None       
        self._run_count += 1    
        
        if self._verbose_level > 0:            
            print("Evaluation metrics:") 
            # print the evaluation metrics
            if self._evaluation == "grid_search":
                for metric, value in validation_performance_scores.items():
                    output = metric + ': ' + ', '.join(str(round(v, 4)) for v in value)
                    print("    " + output)
            else:
                for metric, values in {**validation_performance_scores}.items():
                    output = metric + ': ' \
                                + np.array2string(np.mean(values), precision=4) + ' [std=' \
                                + np.array2string(np.std(values), precision=4) + ']'
                    print("    " + output)

        
        
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


    def save(self, path):
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
    
    def load(self, path):
        """
        Load the pipeline from a file
        
        Parameters
        ----------
        path: string
            The path the pipeline should be loaded from
        """
        self._pipeline = load(path)
        

    # Save predictions to CSV
    def save_predictions(self, prediction_df, path):
        """
        Save predictions to a CSV file
        
        Parameters
        ----------
        prediction_df: pandas.DataFrame
            The data to be predicted
        path: str
            The path to the CSV file the predictions should be saved to
        """
        if not self._is_initialized():
            raise Exception('The pipeline is not initialized yet. Please run the pipeline first.')
        
        predictions = self._pipeline.predict(prediction_df)
        pd.DataFrame(predictions).to_csv(path, index=False)


    def _is_initialized(self):
        """
        Check if the pipeline is initialized
        """
        is_initialized = self._pipeline is not None            
        
        if self._run_count == 0:
            is_initialized = False
            
        return is_initialized
    
    def _do_grid_search(self, param_grid, scoring='accuracy', cv=5, n_jobs=-1):
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
        
        # select the refit value
        refit_value = None
        if isinstance(scoring, dict):
            refit_value = list(scoring.keys())[0]
        else:
            refit_value = scoring        
        
        grid_search = GridSearchCV(self._pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs, refit=refit_value, verbose=self._verbose_level)
        grid_search.fit(self._x_train_df, self._y_train_df)
        self._pipeline = grid_search.best_estimator_
        
        print("Finished performing grid search") if self._verbose_level > 0 else None
        
        # print the best parameters
        if self._verbose_level > 0:
            print("Best parameters:")
            for param, value in grid_search.best_params_.items():
                output = param + ': ' + str(value)
                print("    " + output)
            
        # select validation scores
        validation_performance_scores = {}
        for metric, values in {**grid_search.cv_results_}.items():
            '''if metric.startswith('split'):
                validation_performance_scores[metric] = values'''
            if metric.startswith('mean'):
                validation_performance_scores[metric] = [values[grid_search.best_index_]]
            elif metric.startswith('std'):
                validation_performance_scores[metric] = [values[grid_search.best_index_]]
        # sort the dictionary
        validation_performance_scores = dict(sorted(validation_performance_scores.items()))
        
        return validation_performance_scores