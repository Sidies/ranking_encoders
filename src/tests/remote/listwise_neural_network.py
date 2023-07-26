import warnings
import os
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from datetime import timedelta
from time import time

from src.configuration import load_traindata_for_pointwise
from src.pipeline.neural_network_pipeline import pipeline

train_df = load_traindata_for_pointwise()

start = time()

for cv in range(5):
    print('/////////////////////////////////////////////////////////////')
    print('FOLD ', cv, '\n')

    pipeline(train_df, None, 500)
    
runtime = int(time() - start)
print('\nruntime: ' + str(timedelta(seconds=runtime)) + ' [' + str(runtime) + 's]')