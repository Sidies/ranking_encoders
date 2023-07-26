import warnings
import os
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import pandas as pd
from category_encoders.binary import BinaryEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder
from datetime import timedelta
from sklearn.compose import ColumnTransformer
from time import time

from src import configuration as config
from src.pipeline.pipeline_factory import PipelineFactory, ModelType, EvaluationType
from src.pipeline.pipeline_transformers import *

train_df = config.load_traindata_for_pointwise()
pipelineFactory = PipelineFactory()

start = time()

# running the pipeline plain wihout parameter tuning using cross validation
pipeline = pipelineFactory.create_pipeline(
    train_df,
    ModelType.POINTWISE_CLASSIFICATION_NO_SEARCH,
    evaluation=EvaluationType.CROSS_VALIDATION,
    verbose_level=1,
    n_folds=5,
    workers=16,
    target="rank"
)
pipeline.run()

runtime = int(time() - start)
print('\nruntime: ' + str(timedelta(seconds=runtime)) + ' [' + str(runtime) + 's]')