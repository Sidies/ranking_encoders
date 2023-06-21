import pandas as pd
import argparse
from src import configuration as config
from src.pipeline.model_pipeline import ModelPipeline
from src.pipeline.pipeline_factory import PipelineFactory, ModelType

def main(args):
    
    # get the specific pipeline type
    pipeline_factory = PipelineFactory()

    #x_train = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7], 'y': [4, 5, 6, 7, 8, 9, 10]})
    #y_train = pd.DataFrame({'z': [1, 0, 1, 1, 0, 0, 1]})
    
    x_train, y_train = config.load_traindata_for_regression()
    
    print(y_train.value_counts())

    pipeline = pipeline_factory.create_pipeline(x_train_df=x_train, 
                                                y_train_df=y_train, 
                                                model_type=args.pipeline_type,
                                                verbose_level=1,
                                                evaluation="basic")
    
    pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pipeline_type', type=ModelType, default=ModelType.BASELINE, help='Type of pipeline to run')
    
    args = parser.parse_args()

    main(args)
