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

pipeline = pipelineFactory.create_pipeline(
    train_df,
    ModelType.POINTWISE_NORMALIZED_REGRESSION_NO_SEARCH,
    verbose_level=1,
    evaluation=EvaluationType.CROSS_VALIDATION,
    n_folds=5,
    workers=1,
    target="rank"
)
pipeline.run()

runtime = int(time() - start)
print('\nruntime: ' + str(timedelta(seconds=runtime)) + ' [' + str(runtime) + 's]')