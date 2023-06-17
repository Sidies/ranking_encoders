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
    
    def __init__(self, x_train_df, y_train_df, steps=[], verbose_level=0, evaluation:str="basic", param_grid=[]):
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
            metric = list(performance_metrics.keys())[0]            
            validation_performance_scores = self._do_grid_search(param_grid=self._param_grid, scoring=metric)
            
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
                return # no need to insert if step already exists
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


    def remove_step(self, name:str):
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