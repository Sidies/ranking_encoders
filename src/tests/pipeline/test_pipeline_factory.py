"""
To run this test, run the following command from the root directory:
python -m unittest src.tests.pipeline.test_model_pipeline
"""
import os
import unittest
import pandas as pd
from src import configuration as config
from src.pipeline.pipeline_factory import PipelineFactory, ModelType, EvaluationType

class TestModelPipeline(unittest.TestCase):
    
    def setUp(self) -> None:
        self.pipeline_factory = PipelineFactory()
    
    def test_basic_pipeline(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]})
        pipeline = self.pipeline_factory.create_pipeline(df, target="z", model_type=ModelType.REGRE_BASELINE)
        
        # check that the pipeline has been created by checking that the
        # pipeline has a LinearRegression step
        self.assertTrue('estimator' in pipeline.get_pipeline().named_steps)
        
        
    def test_save_predictions(self):
        train_df = config.load_traindata_for_regression()
        test_df = config.load_testdata_for_regression()
        
        pipeline = self.pipeline_factory.create_pipeline(X_train=train_df, 
                                                         target="cv_score", 
                                                         model_type=ModelType.REGRE_BASELINE,
                                                         X_test=test_df,
                                                         verbose_level=0)
        
        pipeline.run()
        
        # assert whether the predictions have been saved
        save_path = config.DATA_DIR / 'processed/regression_tyrell_prediction.csv'
        self.assertTrue(os.path.exists(save_path))
        
        
    def test_grid_search_pipeline(self):
        df = config.load_traindata_for_regression()
        
        param_grid = {
            "estimator__strategy": ["mean", "median"],
        }
        
        pipeline = self.pipeline_factory.create_pipeline(df, 
                                                         model_type=ModelType.REGRE_BASELINE,
                                                         evaluation=EvaluationType.GRID_SEARCH,
                                                         verbose_level=1,
                                                         param_grid=param_grid,
                                                         n_folds=2,
                                                         workers=2)
        
        pipeline.run()
        
if __name__ == '__main__':
    unittest.main()