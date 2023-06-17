"""
To run this test, run the following command from the root directory:
python -m unittest src.tests.pipeline.test_model_pipeline
"""
import os
import unittest
import pandas as pd
from src import configuration as config
from sklearn.linear_model import LinearRegression
from src.pipeline.model_pipeline import ModelPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class TestModelPipeline(unittest.TestCase):
    
    def setUp(self):
        """
        The setUp method is run before each test
        """
        x_train = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        y_train = pd.DataFrame({'z': [7, 8, 9]})
        self.pipeline = ModelPipeline(x_train, y_train)
        
        '''@self.pipeline.get_pipeline_step_decorator()
        def my_transformer(*args, **kwargs):
            return LinearRegression(*args, **kwargs)

        my_transformer(name="linearregression", position=None)'''
        
        self.pipeline.change_estimator(LinearRegression())        
        self.pipeline.run()
        
        
    def test_pipeline(self):               
        # check that the pipeline has been created by checking that the
        # pipeline has a LinearRegression step
        self.assertTrue('estimator' in self.pipeline.get_pipeline().named_steps)
        
        
    def test_save_predictions(self):
        # create a dataframe that contains data to be predicted
        test_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        project_dir = config.ROOT_DIR
     
        # set the path to the test predictions CSV
        path = os.path.join(project_dir, 'data/testing/test_predictions.csv')
        self.pipeline.save_predictions(test_df, path)
        
        # check that the csv exists
        self.assertTrue(os.path.exists(path))
        
        
    def test_add_new_step(self):
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
        
        
if __name__ == '__main__':
    unittest.main()