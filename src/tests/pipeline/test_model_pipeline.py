"""
To run this test, run the following command from the root directory:
python -m unittest src.tests.pipeline.test_model_pipeline
"""
import os
import unittest
import pandas as pd
from src import configuration as config
from sklearn.linear_model import LinearRegression
from src.pipeline.model_pipeline import ModelPipeline, EvaluationType
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor, DummyClassifier
from src.pipeline.pipeline_transformers import PrintDataframe, GroupDataframe, TextPrinter
from src.pipeline.evaluation.evaluate_regression import BaseShuffleSplit, custom_train_test_split

class TestModelPipeline(unittest.TestCase):
    
    def setUp_classification(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]})
        self.pipeline = ModelPipeline(df, split_factors=[], target="z", evaluation=EvaluationType.BASIC)
        
        '''@self.pipeline.get_pipeline_step_decorator()
        def my_transformer(*args, **kwargs):
            return LinearRegression(*args, **kwargs)

        my_transformer(name="linearregression", position=None)'''
        
        self.pipeline.change_estimator(DummyClassifier())        
        self.pipeline.run()
        
        
    def test_pipeline(self):          
        self.setUp_classification()     
        # check that the pipeline has been created by checking that the
        # pipeline has a LinearRegression step
        self.assertTrue('estimator' in self.pipeline.get_pipeline().named_steps)
        
        
    def test_add_new_step(self):
        self.setUp_classification()
        self.pipeline.add_new_step(StandardScaler(), "scaler")
        
        # test if step has been added
        self.assertTrue('scaler' in self.pipeline.get_pipeline().named_steps)
        
        # add another step and test if it has been added
        self.pipeline.add_new_step_at_position(SimpleImputer(strategy="most_frequent"), "imputer", 0)
        
        # test if step has been added at position 0
        self.assertEqual(list(self.pipeline.get_pipeline().named_steps.keys())[0], 'imputer')
        
        # now remove a step and check if it has been removed
        self.pipeline.remove_step("imputer")
        self.assertFalse('imputer' in self.pipeline.get_pipeline().named_steps)
        
        
    def test_regression_pipeline(self):
        df = config.load_traindata_for_regression()
        self.pipeline = ModelPipeline(df, evaluation=EvaluationType.BASIC)
        self.pipeline.clear_steps()
        '''@self.pipeline.get_pipeline_step_decorator()
        def my_transformer(*args, **kwargs):
            return LinearRegression(*args, **kwargs)

        my_transformer(name="linearregression", position=None)'''
        self.pipeline.change_estimator(DummyRegressor())
        self.pipeline.run()
        
        
    def test_cross_validate(self):
        df = config.load_traindata_for_regression()
        
        self.pipeline = ModelPipeline(df, evaluation=EvaluationType.CROSS_VALIDATION, verbose_level=1, n_folds=1)
        self.pipeline.clear_steps()
        # self.pipeline.add_new_step(PrintDataframe(verbose=1), "print_df_1")
        # self.pipeline.add_new_step(TextPrinter(f"Starting to group by factors: {self.pipeline._split_factors}", verbose=1), "print_text_1")
        # self.pipeline.add_new_step(GroupDataframe(groupby=self.pipeline._split_factors), "group_df")
        #self.pipeline.add_new_step(PrintDataframe(verbose=1), "print_df_2")
        
        self.pipeline.change_estimator(DummyRegressor())
        self.pipeline.run()
        
        
        
if __name__ == '__main__':
    unittest.main()