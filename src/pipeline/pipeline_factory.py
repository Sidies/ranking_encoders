import pandas as pd
from src.pipeline.model_pipeline import ModelPipeline
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from enum import Enum

class ModelType(Enum):
    BASELINE = 'Baseline'
    

class PipelineFactory:
    """
    Class that creates ModelPipelines
    """
    def create_pipeline(self, x_train_df:pd.DataFrame, y_train_df, model_type:ModelType, **kwargs):
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
        if model_type == ModelType.BASELINE:
            pipeline_steps = [
                ("estimator", DummyClassifier(strategy="most_frequent"))    
            ]   
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")            
            
        # create the new pipeline and return it
        return ModelPipeline(x_train_df, 
                             y_train_df, 
                             steps=pipeline_steps, 
                             **kwargs)

        