"""
This module consists of necessary configuration data used in other parts of the project.
"""
import os
import pandas as pd
from pathlib import Path
import warnings
from typing import Union
from src.features.pairwise_utils import prepare_data

def ignore_warnings():
    """
    Ignore warnings that are generated by the sklearn library.
    """
    warnings.filterwarnings("ignore")


# Static variables that contain paths to directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
FIGURE_DIR = ROOT_DIR / "reports/figures"
DATA_PROCESSED_DIR = DATA_DIR / "processed"


def print_divider_line():
    # Prints a line consisting of 40 dashes
    print('-' * 40)


def load_dataset(path: Union[Path, str]) -> pd.DataFrame:
    return pd.read_csv(path)


def load_rankings(path: Union[Path, str]) -> pd.DataFrame:
    out = pd.read_csv(path, index_col=0, header=[0, 1, 2, 3])
    out.columns.name = ("dataset", "model", "tuning", "scoring")
    return out


def load_whole_data():
    x_train_df = load_dataset(os.path.join(ROOT_DIR, 'data/raw/dataset.csv'))
    y_train_df = load_rankings(os.path.join(ROOT_DIR, 'data/raw/rankings.csv'))
    merged_df = pd.concat([x_train_df, y_train_df], axis=1)
    return merged_df, x_train_df, y_train_df


def load_traindata_for_regression():
    return load_dataset(os.path.join(ROOT_DIR, 'data/raw/dataset_train.csv'))


def load_traindata_for_pointwise():
    return load_dataset(os.path.join(ROOT_DIR, 'data/raw/dataset_rank_train.csv'))


def load_traindata_for_pairwise():
    df_train = load_dataset(os.path.join(ROOT_DIR, 'data/raw/dataset_rank_train.csv'))
    X_train, X_test, y_train = prepare_data(df_train, factors=["dataset", "model", "tuning", "scoring"], target="rank", column_to_compare="encoder")
    return X_train, X_test, y_train


def load_testdata_for_regression():
    return load_dataset(os.path.join(ROOT_DIR, 'data/raw/X_test.csv'))