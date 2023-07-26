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
from src.features.encoder_utils import NoY
from src.pipeline.pipeline_factory import PipelineFactory, ModelType, EvaluationType
from src.pipeline.pipeline_transformers import *


# load the data
df_train = config.load_traindata_for_pointwise()
pipelineFactory = PipelineFactory()

start = time()

pipeline = pipelineFactory.create_pipeline(
    train_df=df_train,
    model_type=ModelType.PAIRWISE_CLASSIFICATION_NO_SEARCH,
    verbose_level=1,
    evaluation=EvaluationType.CROSS_VALIDATION,
    target="rank",
    as_pairwise=True
)
pipeline.run()

runtime = int(time() - start)
print('\nruntime: ' + str(timedelta(seconds=runtime)) + ' [' + str(runtime) + 's]')