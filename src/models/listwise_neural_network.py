import pandas as pd
import numpy as np
from src import configuration as config
from src.pipeline.evaluation.evaluation_utils import custom_train_test_split
import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
import array
import collections
import keras

from typing import Dict, List, Optional, Text, Tuple

def _create_feature_dict() -> Dict[Text, List[tf.Tensor]]:
  """Helper function for creating an empty feature dict for defaultdict."""
  return {"encoder": [], "rank": []}


def _sample_list(
    feature_lists: Dict[Text, List[tf.Tensor]],
    num_examples_per_list: int,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Function for sampling a list example from given feature lists."""
  if random_state is None:
    random_state = np.random.RandomState()

  sampled_indices = random_state.choice(
      range(len(feature_lists["encoder"])),
      size=num_examples_per_list,
      replace=False,
  )
  sampled_movie_titles = [
      feature_lists["encoder"][idx] for idx in sampled_indices
  ]
  sampled_ratings = [
      feature_lists["rank"][idx]
      for idx in sampled_indices
  ]

  return (
      tf.stack(sampled_movie_titles, 0),
      tf.stack(sampled_ratings, 0),
  )


def sample_listwise(
    rating_dataset: tf.data.Dataset,
    num_list_per_feature: int = 32,
    num_examples_per_list: int = 32,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
  """Function for converting the MovieLens 100K dataset to a listwise dataset.

  Args:
      rating_dataset:
        The MovieLens ratings dataset loaded from TFDS with features
        "movie_title", "user_id", and "user_rating".
      num_list_per_user:
        An integer representing the number of lists that should be sampled for
        each user in the training dataset.
      num_examples_per_list:
        An integer representing the number of movies to be sampled for each list
        from the list of movies rated by the user.
      seed:
        An integer for creating `np.random.RandomState`.

  Returns:
      A tf.data.Dataset containing list examples.

      Each example contains three keys: "user_id", "movie_title", and
      "user_rating". "user_id" maps to a string tensor that represents the user
      id for the example. "movie_title" maps to a tensor of shape
      [sum(num_example_per_list)] with dtype tf.string. It represents the list
      of candidate movie ids. "user_rating" maps to a tensor of shape
      [sum(num_example_per_list)] with dtype tf.float32. It represents the
      rating of each movie in the candidate list.
  """
  random_state = np.random.RandomState(seed)

  example_lists_by_user = collections.defaultdict(_create_feature_dict)

  movie_title_vocab = set()
  for example in rating_dataset:
    user_id = example["features"].numpy()
    example_lists_by_user[user_id]["encoder"].append(
        example["encoder"])
    example_lists_by_user[user_id]["rank"].append(
        example["rank"])
    movie_title_vocab.add(example["encoder"].numpy())

  tensor_slices = {"features": [], "encoder": [], "rank": []}

  for user_id, feature_lists in example_lists_by_user.items():
    for _ in range(num_list_per_feature):

      # Drop the user if they don't have enough ratings.
      if len(feature_lists["encoder"]) < num_examples_per_list:
        continue

      sampled_movie_titles, sampled_ratings = _sample_list(
          feature_lists,
          num_examples_per_list,
          random_state=random_state,
      )
      tensor_slices["features"].append(user_id)
      tensor_slices["encoder"].append(sampled_movie_titles)
      tensor_slices["rank"].append(sampled_ratings)

  return tf.data.Dataset.from_tensor_slices(tensor_slices)


def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


class RankingModel(tfrs.Model):

    def __init__(self, loss, unique_factor_combinations, unique_encoder_rankings):
        super().__init__()
        embedding_dimension = 32
        # Compute embeddings for factor combinations.
        self.factors_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
            vocabulary=unique_factor_combinations),
            tf.keras.layers.Embedding(len(unique_factor_combinations) + 2, embedding_dimension)
        ])
        
        # Compute embeddings for encoder combinations.
        self.encoder_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_encoder_rankings),
            tf.keras.layers.Embedding(len(unique_encoder_rankings) + 2, embedding_dimension)
        ])

        # Compute predictions.
        self.score_model = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ])

        self.task = CustomRanking(
            loss=loss,
            metrics=[
                tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
                tf.keras.metrics.RootMeanSquaredError()
            ]
        )

    def call(self, features):
        # We first convert the id features into embeddings.
        # User embeddings are a [batch_size, embedding_dim] tensor.
        user_embeddings = self.factors_embeddings(features["features"])

        # Movie embeddings are a [batch_size, num_movies_in_list, embedding_dim]
        # tensor.
        movie_embeddings = self.encoder_embeddings(features["encoder"])

        # We want to concatenate user embeddings with movie emebeddings to pass
        # them into the ranking model. To do so, we need to reshape the user
        # embeddings to match the shape of movie embeddings.
        #list_length = features["encoder"].shape[1]
        # get list length for my shape (10,) tensor
        list_length = features["encoder"].shape[1]
        user_embedding_repeated = tf.repeat(
            tf.expand_dims(user_embeddings, 1), [list_length], axis=1)

        # Once reshaped, we concatenate and pass into the dense layers to generate
        # predictions.
        concatenated_embeddings = tf.concat(
            [user_embedding_repeated, movie_embeddings], 2)

        return self.score_model(concatenated_embeddings)

    def compute_loss(self, features, training=False):
        labels = features.pop("rank")

        scores = self(features)

        return self.task(
            labels=labels,
            predictions=tf.squeeze(scores, axis=-1),
        )
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'factors_embeddings': self.factors_embeddings,
            'encoder_embeddings': self.encoder_embeddings,
            'score_model': self.score_model,
            'task': self.task
        })
        return config
    
    
class CustomRanking(tfrs.tasks.Ranking):
    def __init__(self, loss, **kwargs):
        super().__init__(loss=loss, **kwargs)

    def get_config(self):
        config = super().get_config()
        return config
    
def revert_target(original_data, transformed_data, prediction):
    # transform data into dictionary
    transformed_data = {
        col: list(transformed_data)[0][col].numpy() for col in list(transformed_data)[0]
    }

    # revert factor concatenation
    split_factors = {'dataset': [], 'model': [], 'tuning': [], 'scoring': []}
    for feature in transformed_data['features']:
        split_factor = feature.split()
        split_factors['dataset'].append(split_factor[0])
        split_factors['model'].append(split_factor[1])
        split_factors['tuning'].append(split_factor[2])
        split_factors['scoring'].append(split_factor[3])
    transformed_data.pop('features')
    transformed_data = {**split_factors, **transformed_data}

    # zip encoders and rankings together
    reverted_data = {'dataset': [], 'model': [], 'tuning': [], 'scoring': [], 'encoder': [], 'rank_pred': []}
    for row_index in range(len(next(iter(transformed_data.values())))):
        row = {key: transformed_data[key][row_index] for key in transformed_data}
        encoder_rankings = zip(row['encoder'], prediction[row_index])
        for encoder, rank in encoder_rankings:
            reverted_data['dataset'].append(row['dataset'].decode('utf-8'))
            reverted_data['model'].append(row['model'].decode('utf-8'))
            reverted_data['tuning'].append(row['tuning'].decode('utf-8'))
            reverted_data['scoring'].append(row['scoring'].decode('utf-8'))
            reverted_data['encoder'].append(encoder.decode('utf-8'))
            reverted_data['rank_pred'].append(rank[0])
    reverted_data = pd.DataFrame(reverted_data)

    reverted_data[['dataset', 'model', 'tuning', 'scoring', 'encoder']] = reverted_data[['dataset', 'model', 'tuning', 'scoring', 'encoder']].astype(original_data[[
        'dataset', 'model', 'tuning', 'scoring', 'encoder'
    ]].dtypes)

    reverted_data = original_data.merge(
        reverted_data, 
        on=['dataset', 'model', 'tuning', 'scoring', 'encoder'],
        how='outer',
    )
    reverted_data.index = original_data.index

    return reverted_data['rank_pred'].fillna(0)
