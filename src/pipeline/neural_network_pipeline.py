import pandas as pd
import numpy as np
from src import configuration as config
from src.pipeline.evaluation.evaluation_utils import average_spearman, custom_train_test_split, get_rankings
from src.models.listwise_neural_network import sample_listwise, RankingModel, revert_target
import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs

def pipeline(train_df, X_test=None, epochs=10):
    
    print("Starting Neural Network Pipeline")
    
    # load the data train data
    df = train_df  
    # if df contains cv_score drop it
    if 'cv_score' in df.columns:
        df = df.drop(columns=['cv_score'])
    
    # load the test data
    if X_test is not None:
        df_test = X_test
        if 'cv_score' in df.columns:
            df_test = df_test.drop(columns=['cv_score'])
        df_test['rank'] = np.full(len(df_test), 0)
    else:
        # do train test split if no test data is provided
        X_train, X_val, y_train, y_val = custom_train_test_split(df, factors=["dataset", "model", "tuning", "scoring"], target="rank")
        df = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_val, y_val], axis=1)
    
    # prepare the data
    # train data
    df_prep = df.copy()
    df_prep['dataset'] = df_prep['dataset'].astype(str)
    df_prep['features'] = df_prep['dataset'].astype(str) + ' ' + df_prep['model'] + ' ' + df_prep['tuning'] + ' ' + df_prep['scoring']
    df_prep = df_prep.drop(columns=['dataset', 'model', 'tuning', 'scoring'])


    # test data
    df_test_prep = df_test.copy()
    df_test_prep['dataset'] = df_test_prep['dataset'].astype(str)
    df_test_prep['features'] = df_test_prep['dataset'].astype(str) + ' ' + df_test_prep['model'] + ' ' + df_test_prep['tuning'] + ' ' + df_test_prep['scoring']
    df_test_prep = df_test_prep.drop(columns=['dataset', 'model', 'tuning', 'scoring'])
    
    # load to tensor
    df_tf = tf.data.Dataset.from_tensor_slices(dict(df_prep))
    df_tf_test = tf.data.Dataset.from_tensor_slices(dict(df_test_prep))
    
    # sample listwise
    print("Sampling listwise")
    df_listwise = sample_listwise(df_tf, 1, 32)
    df_listwise_test = sample_listwise(df_tf_test, 1, 32)
    cached_train = df_listwise.shuffle(100_000).batch(8192).cache()
    cached_test = df_listwise_test.batch(4096).cache()
    
    # prepare vocabulary
    unique_factor_combinations = np.unique(df_prep[['features']])
    unique_factor_combinations = unique_factor_combinations.astype('S')

    unique_encoder_rankings = np.unique(df_prep[['encoder']])
    unique_encoder_rankings = unique_encoder_rankings.astype('S')
    
    # prepare and run the model
    listwise_model = RankingModel(tfr.keras.losses.ListMLELoss(), unique_factor_combinations, unique_encoder_rankings)
    listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
    listwise_model.fit(cached_train, epochs=epochs, verbose=True)
    
    if X_test is None:
        # Evaluate the model
        print("Evaluating the model")    
        listwise_model_result = listwise_model.evaluate(cached_test, return_dict=True)
        print("NDCG of the MSE Model: {:.4f}".format(listwise_model_result["ndcg_metric"]))

        df = pd.concat([X_val, y_val, revert_target(df_test, cached_test, listwise_model.predict(cached_test))], axis=1)
        y_true_rankings = get_rankings(
            df=df,
            factors=['dataset', 'model', 'tuning', 'scoring'],
            new_index='encoder',
            target='rank'
        )
        y_pred_rankings = get_rankings(
            df=df,
            factors=['dataset', 'model', 'tuning', 'scoring'],
            new_index='encoder',
            target='rank_pred'
        )
        print("Average Spearman of the MSE Model: {:.4f}".format(average_spearman(y_true_rankings, y_pred_rankings)))
    else:
        # save the predictions to csv if test data is provided    
        print("Saving the predictions to csv")
        prediction = listwise_model.predict(cached_test)
        prediction = revert_target(df_test, cached_test, prediction)
        prediction.to_csv(config.DATA_PROCESSED_DIR / "listwise_tyrell_prediction.csv", index=False, header=False)
        