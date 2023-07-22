import pandas as pd
import numpy as np
from src import configuration as config
from src.pipeline.evaluation.evaluation_utils import custom_train_test_split
from src.models.listwise_neural_network import sample_listwise, RankingModel
import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs

def pipeline(train_df, test_df=None, epochs=10):
    
    print("Starting Neural Network Pipeline")
    
    # load the data train data
    df = train_df  
    # if df contains cv_score drop it
    if 'cv_score' in df.columns:
        df = df.drop(columns=['cv_score'])
    
    # load the test data
    if test_df is not None:
        df_test = test_df
        if 'cv_score' in df.columns:
            df_test = df_test.drop(columns=['cv_score'])
    else:
        # do train test split if no test data is provided
        X_train, X_test, y_train, y_test = custom_train_test_split(df, factors=["dataset", "model", "tuning", "scoring"], target="rank")
        df = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)
    
    # prepare the data
    # train data
    df['dataset'] = df['dataset'].astype(str)
    df['features'] = df['dataset'].astype(str) + ' ' + df['model'] + ' ' + df['tuning'] + ' ' + df['scoring']
    df = df.drop(columns=['dataset', 'model', 'tuning', 'scoring'])


    # test data
    df_test['dataset'] = df_test['dataset'].astype(str)
    df_test['features'] = df_test['dataset'].astype(str) + ' ' + df_test['model'] + ' ' + df_test['tuning'] + ' ' + df_test['scoring']
    df_test = df_test.drop(columns=['dataset', 'model', 'tuning', 'scoring'])
    
    # load to tensor
    df_tf = tf.data.Dataset.from_tensor_slices(dict(df))
    df_tf_test = tf.data.Dataset.from_tensor_slices(dict(df_test))
    
    # sample listwise
    print("Sampling listwise")
    df_listwise = sample_listwise(df_tf)
    df_listwise_test = sample_listwise(df_tf_test)
    cached_train = df_listwise.shuffle(100_000).batch(8192).cache()
    cached_test = df_listwise_test.batch(4096).cache()
    
    # prepare vocabulary
    unique_factor_combinations = np.unique(df[['features']])
    unique_factor_combinations = unique_factor_combinations.astype('S')

    unique_encoder_rankings = np.unique(df[['encoder']])
    unique_encoder_rankings = unique_encoder_rankings.astype('S')
    
    # prepare and run the model
    listwise_model = RankingModel(tfr.keras.losses.ListMLELoss(), unique_factor_combinations, unique_encoder_rankings)
    listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
    listwise_model.fit(cached_train, epochs=epochs, verbose=True)
    
    # Evaluate the model
    print("Evaluating the model")    
    listwise_model_result = listwise_model.evaluate(cached_test, return_dict=True)
    print("NDCG of the MSE Model: {:.4f}".format(listwise_model_result["ndcg_metric"]))
    
    # save the predictions to csv if test data is provided
    if test_df is not None:
        print("Saving the predictions to csv")
        predictions = listwise_model.predict(cached_test)
        predictions.to_csv(config.DATA_PROCESSED_DIR / "listwise_tyrell_prediction.csv", index=False)