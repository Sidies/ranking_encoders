import pandas as pd
import argparse
from src import configuration as config
from src.pipeline.model_pipeline import ModelPipeline, EvaluationType
from src.pipeline.pipeline_factory import PipelineFactory, ModelType

def main(args):
    
    # get the specific pipeline type
    pipeline_factory = PipelineFactory()
    dataset_path = config.ROOT_DIR / 'data/raw/' / args.dataset
    
    train_df = config.load_dataset(dataset_path)

    pipeline = pipeline_factory.create_pipeline(train_df, 
                                                model_type=args.pipeline_type,
                                                verbose_level=1,
                                                evaluation=EvaluationType.BASIC)
    
    pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pipeline_type', type=str, default='regre_baseline', help='Type of pipeline to run')
    parser.add_argument('--dataset', type=str, default='dataset_train.csv', help='Dataset name to use for training')
    
    args = parser.parse_args()

    main(args)
