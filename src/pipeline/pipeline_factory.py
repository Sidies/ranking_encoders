import pandas as pd
from src.pipeline.model_pipeline import ModelPipeline, EvaluationType
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from enum import Enum

class ModelType(Enum):
    REGRE_BASELINE = "regre_baseline"
    CLASS_BASELINE = "class_baseline"
     

class PipelineFactory:
    """
    Class that creates ModelPipelines
    """
    def create_pipeline(self, X_train:pd.DataFrame, 
                        model_type:ModelType, 
                        evaluation:EvaluationType=EvaluationType.BASIC, 
                        y_train=None, 
                        X_test=None, 
                        verbose_level=0, 
                        **kwargs):
        """
        Create a ModelPipeline of the specified type.

        Args:
            model_type (ModelType): The type of ModelPipeline to create
            x_train_df (pd.DataFrame): The training data
            y_train_df (pd.DataFrame): The training labels
            verbose (int, optional): The verbosity level. Defaults to 0.
            evaluation (str, optional): The type of evaluation to perform. 
                Can be one of "basic", "cross_validation", "grid_search". Defaults to "basic".

        Raises:
            ValueError: If the model_type is not recognized

        Returns:
            ModelPipeline: The created ModelPipeline
        """
        if y_train is not None:
            # merge the X and y dataframes
            X_train = pd.concat([X_train, y_train], axis=1)
        
        if model_type == "regre_baseline" or model_type == ModelType.REGRE_BASELINE:
            pipeline_steps = [
                ("estimator", DummyRegressor(strategy="mean"))    
            ]   
            
        elif model_type == "class_baseline" or model_type == ModelType.CLASS_BASELINE:
            pipeline_steps = [
                ("estimator", DummyClassifier(strategy="most_frequent"))
            ]
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")            
            
        # create the new pipeline and return it
        return ModelPipeline(X_train, 
                             steps=pipeline_steps,
                             evaluation=evaluation,
                             X_test=X_test,
                             verbose_level=verbose_level,
                             **kwargs)
        
