import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import argparse

from src import configuration as config
from src.pipeline.model_pipeline import EvaluationType
from src.pipeline.pipeline_factory import PipelineFactory
from src.pipeline import neural_network_pipeline


def run_pipeline(args):
    """
    Takes the arguments, creates and runs a pipeline.

    Args:
        args : The arguments to use for creating the pipeline
    """

    # pipeline factory thas is able to create different pipeline by type
    pipeline_factory = PipelineFactory()

    # load the data
    train_df_path = config.DATA_RAW_DIR / args.train_dataset
    train_df = config.load_dataset(train_df_path)
    if args.y_train_dataset != '':
        y_train_df_path = config.DATA_RAW_DIR / args.y_train_dataset
        y_train_df = config.load_dataset(y_train_df_path)

        # merge the X and y dataframes
        if args.target in train_df.columns:
            train_df = train_df.drop(columns=[args.target])
        train_df = pd.concat([train_df, y_train_df], axis=1)

    test_df = None
    if args.test_dataset != '':
        test_df_path = config.DATA_RAW_DIR / args.test_dataset
        test_df = config.load_dataset(test_df_path)

    pipeline = pipeline_factory.create_pipeline(
        train_df=train_df,
        model_type=args.pipeline_type,
        verbose_level=1,
        evaluation=EvaluationType.CROSS_VALIDATION,
        X_test=test_df,
        target=args.target,
    )

    pipeline.run()
    
    
def run_neural_network(args):
    
    # load the data
    train_df_path = config.DATA_RAW_DIR / args.train_dataset
    train_df = config.load_dataset(train_df_path)
    if args.y_train_dataset != '':
        y_train_df_path = config.DATA_RAW_DIR / args.y_train_dataset
        y_train_df = config.load_dataset(y_train_df_path)

        # merge the X and y dataframes
        if args.target in train_df.columns:
            train_df = train_df.drop(columns=[args.target])
        train_df = pd.concat([train_df, y_train_df], axis=1)

    test_df = None
    if args.test_dataset != '':
        test_df_path = config.DATA_RAW_DIR / args.test_dataset
        test_df = config.load_dataset(test_df_path)
    
    neural_network_pipeline.pipeline(train_df, test_df, args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pipeline_type', type=str, default='pairwise_classification_optuna_search', help='Type of pipeline to run')
    parser.add_argument(
        '--train_dataset',
        type=str,
        default='dataset_rank_train.csv',
        help='Dataset name to use for training'
    )
    parser.add_argument('--test_dataset', type=str, default='', help='Dataset name to use for testing')
    parser.add_argument('--y_train_dataset', type=str, default='', help='Dataset name to use for training labels')
    parser.add_argument('--target', type=str, default='rank', help='Target column name')
    parser.add_argument('--as_neural_network', type=bool, default=True, help='Whether to run as neural network')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to run')
    args = parser.parse_args()

    if args.as_neural_network:
        run_neural_network(args)
    else:
        run_pipeline(args)
