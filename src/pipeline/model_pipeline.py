import pandas as pd
import numpy as np
import enum
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.base import is_classifier, is_regressor
from src.pipeline.evaluation import evaluate_regression
from src import configuration as config

class EvaluationType(enum.Enum):
    BASIC = "basic"
    CROSS_VALIDATION = "cross_validation"
    GRID_SEARCH = "grid_search"

class ModelPipeline:
    """
    Class that handles the data pipeline
    """
    _run_count:int = 0
    
    def __init__(self, 
                 df:pd.DataFrame,
                 target:str ="cv_score",
                 steps=[], 
                 verbose_level=0, 
                 evaluation:EvaluationType=EvaluationType.BASIC,
                 X_test:pd.DataFrame=None,                  
                 split_factors=["dataset", "model", "tuning", "scoring"],
                 param_grid=[]):
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
        """        
        self._pipeline = Pipeline(steps=steps)        
        self._df = df
        self._target = target
        self._evaluation = evaluation
        self._verbose_level = verbose_level
        self.X_test = X_test
        self._split_factors = split_factors
        self._param_grid = param_grid
            
    
    # start the pipeline
    def run(self):
        
        print(f"Starting pipeline using method: {self._evaluation}") if self._verbose_level > 0 else None
        
        validation_performance_scores = {}
        
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
                X_train, X_test, y_train, y_test = evaluate_regression.custom_train_test_split(self._df, self._split_factors, self._target, train_size=0.75, random_state=42)
            
            if not is_regression:
                self._pipeline.fit(X_train, y_train)
                y_pred = self._pipeline.predict(X_test)
                validation_performance_scores['validation_accuracy'] = [accuracy_score(y_test, y_pred)]
                validation_performance_scores['validation_f1-score'] = [f1_score(y_test, y_pred, average='macro')]
                validation_performance_scores['validation_mcc'] = [matthews_corrcoef(y_test, y_pred)]
                
            elif is_regression:
                new_index = "encoder"
                y_pred = pd.Series(self._pipeline.fit(X_train, y_train).predict(X_test), index=y_test.index, name="cv_score_pred")
                df_pred = pd.concat([X_test, y_test, y_pred], axis=1)
                # ---- convert to rankings and evaluate
                rankings_test = evaluate_regression.get_rankings(df_pred, factors=self._split_factors, new_index=new_index, target="cv_score")
                rankings_pred = evaluate_regression.get_rankings(df_pred, factors=self._split_factors, new_index=new_index, target="cv_score_pred")
                print(evaluate_regression.average_spearman(rankings_test, rankings_pred))
                
            
        elif self._evaluation == EvaluationType.CROSS_VALIDATION:
            X_train, y_train = self._split_target(self._df, self._target)
            self._pipeline.fit(X_train, y_train)
            validation_performance_scores = cross_validate(
                self._pipeline,
                X_train,
                y_train,
                scoring=performance_metrics,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            )
            
        elif self._evaluation == EvaluationType.GRID_SEARCH:
            metric = list(performance_metrics.keys())[0]        
            X_train, y_train = self._split_target(self._df, self._target)    
            validation_performance_scores = self._do_grid_search(X_train, y_train, param_grid=self._param_grid, scoring=metric)
            
        else:
            raise Exception(f"Unknown evaluation type: {self._evaluation}")
            
        print("Finished running the pipeline") if self._verbose_level > 0 else None       
        self._run_count += 1    
        
        # print out the evaluation metrics
        if self._verbose_level > 0 and validation_performance_scores != {}:            
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
        
        
    def save_prediction(self, data:pd.DataFrame):
        """
        Save the predictions to a file at location data/processed.

        Args:
            data : The data to make predictions on
        """
        
        # check if df contains the target column
        if self._target in data.columns:
            data = data.drop(self._target, axis=1)
            
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
    
    def _do_grid_search(self, X_train:pd.DataFrame, y_train, param_grid, scoring='accuracy', cv=5, n_jobs=-1):
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
        grid_search.fit(X_train, y_train)
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