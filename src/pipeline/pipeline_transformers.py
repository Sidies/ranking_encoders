import pandas as pd
import numpy as np
import copy

from category_encoders.binary import BinaryEncoder
from category_encoders import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src import configuration as config
from src.features.embeddings import GraphEmbedding
from src.features.encoder_utils import get_metafeatures


class DebugTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that prints the data it receives.
    """

    def __init__(self, verbose=0):
        self.verbose = verbose

    def transform(self, X):
        if self.verbose > 0:
            if hasattr(X, 'toarray'):  # check if X is a sparse matrix
                df = pd.DataFrame(X.toarray())
                # print(tabulate(df.columns, headers=df.columns, tablefmt='simple'))
                print(df.columns)
            else:
                df = pd.DataFrame(X)
                # print(tabulate(df.columns, headers=df.columns, tablefmt='simple'))
                print(df.columns)

        return X

    def fit(self, X, y=None, **fit_params):
        return self


class PrintDataframe(BaseEstimator, TransformerMixin):
    """
    Transformer that prints the dataframe it receives.
    """

    def __init__(self, verbose=0):
        self.verbose = verbose

    def transform(self, X):
        if self.verbose > 0 and isinstance(X, pd.DataFrame):
            config.print_divider_line()
            print("Printing dataframe:")
            print(X.head())
            config.print_divider_line()

        return X

    def fit(self, X, y=None, **fit_params):
        return self


class TextPrinter(BaseEstimator, TransformerMixin):
    """
    Transformer that prints the text it receives.
    """

    def __init__(self, text, verbose=0):
        self.text = text
        self.verbose = verbose

    def transform(self, X):
        if self.verbose > 0:
            print(f"{self.text}")
        return X

    def fit(self, X, y=None):
        if self.verbose > 0:
            print(f"{self.text}")
        return self


class ColumnKeeper(BaseEstimator, TransformerMixin):
    """
    Transformer that keeps only the specified columns.
    """

    def __init__(self, columns):
        self.columns = columns

    def transform(self, X):
        return X[self.columns]

    def fit(self, X, y=None):
        return self
  
    
class GroupDataframe(BaseEstimator, TransformerMixin):
    def __init__(self, groupby):
        self.factors = groupby

    def transform(self, X):
        X_factors = X.groupby(self.factors).agg(lambda a: np.nan).reset_index()[self.factors]
        X = pd.merge(X_factors, X, on=self.factors)
        return X

    def fit(self, X, y=None):
        return self


# Custom Transformer that drops columns passed as argument
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assuming X is a DataFrame
        return X.drop(self.cols_to_drop, axis=1)


class Node2VecEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, graph,dimensions=2, walk_length=20, num_walks=1000, workers=1, **kwargs):
        self.graph = graph
        self.kwargs = kwargs
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        graph_embedding = GraphEmbedding(self.graph)
        model = graph_embedding.node2vec(
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers,
            **self.kwargs
        )

        # Get the embeddings.
        n2v_embeddings = {node: model.wv.get_vector(node) for node in model.wv.index_to_key}

        X['node2vec_embedding_dim1'] = 0
        X['node2vec_embedding_dim2'] = 0

        for i, row in X.iterrows():
            node = row['encoder']
            embedding = n2v_embeddings[node]
            X.at[i, 'node2vec_embedding_dim1'] = embedding[0]
            X.at[i, 'node2vec_embedding_dim2'] = embedding[1]

        X = X.drop(columns=['encoder'])

        return X


class Node2VecGraphEmbeddingWithKMeans(BaseEstimator, TransformerMixin):
    def __init__(self, graph, dimensions=2, walk_length=20, num_walks=1000, workers=1, **kwargs):
        self.graph = graph
        self.kwargs = kwargs
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        graph_embedding = GraphEmbedding(self.graph)
        model = graph_embedding.node2vec(
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers,
            **self.kwargs
        )

        # Get the embeddings.
        node2vec_embeddings = {node: model.wv.get_vector(node) for node in model.wv.index_to_key}

        # Convert the embeddings to a numpy array.
        embed_array = np.array(list(node2vec_embeddings.values())).astype('double')

        # Use KMeans clustering to cluster the embeddings.
        kmeans = KMeans(n_clusters=6, random_state=0).fit(embed_array)

        # Add a new column to the train_df dataframe for the cluster assignments.
        X['encoder_cluster'] = 0

        # Loop over the rows in the dataframe and assign the cluster labels.
        for i, row in X.iterrows():
            node = row['encoder']
            embedding = node2vec_embeddings[node]
            cluster = kmeans.predict([embedding])[0]
            X.at[i, 'encoder_cluster'] = cluster

        X = X.drop(columns=['encoder'])

        return X


class PoincareEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, graph, epochs=100, batch_size=10, size=2, negative=2, alpha=0.1, encoder=None, **kwargs):
        self.graph = graph
        self.kwargs = kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.size = size
        self.negative = negative
        self.alpha = alpha
        self.encoder = encoder

    def fit(self, X, y=None):
        graph_embedding = GraphEmbedding(self.graph)
        model = graph_embedding.poincare(
            epochs=self.epochs,
            batch_size=self.batch_size,
            size=self.size,
            negative=self.negative,
            alpha=self.alpha,
            **self.kwargs
        )

        # Get the embeddings.
        self._poincare_embeddings = {node: model.kv.get_vector(node) for node in model.kv.index_to_key}

        if self.encoder:
            try:
                self.encoder.fit(X['encoder'], y)
            except Exception:
                try:
                    self.encoder.fit(X['encoder'], None)
                except Exception as e:
                    print('Encoder requires the target, but the target has an invalid format.')
                    raise

        return self

    def transform(self, X):
        X_new = X.copy()
        X_new['poincare_embedding_dim1'] = 0
        X_new['poincare_embedding_dim2'] = 0

        for i, row in X_new.iterrows():
            node = row['encoder']
            embedding = self._poincare_embeddings[node]
            X_new.at[i, 'poincare_embedding_dim1'] = embedding[0]
            X_new.at[i, 'poincare_embedding_dim2'] = embedding[1]

        X_new = X_new.drop(columns=['encoder'])
        if self.encoder:
            X_new = pd.concat([
                X_new,
                self.encoder.transform(X['encoder'])
            ], axis=1)

        return X_new


def get_features_to_drop_by_nan_threshold(df, threshold):
    nan_count = df.isna().sum()
    total_count = nan_count + df.notna().sum()

    rel_nan_count = pd.DataFrame(nan_count / total_count).reset_index()
    features_to_drop = rel_nan_count[rel_nan_count[0] >= threshold]['index']

    return features_to_drop


class OpenMLMetaFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that maps processed OpenML metafeatures to their corresponding dataset.

    Parameters
    ----------
    nan_ratio_feature_drop_threshold: float
        Features with a 'nan' ratio above this value are dropped.
    imputer: sklearn.impute.SimpleImputer
        The imputer used for imputing 'nan' values remaining after the feature dropping.
    scaler: sklearn.preprocessing.StandardScaler
        The scaler used for scaling the values as preparation for the principal component analysis.
    expected_pca_variance: float
        The expected variance that should be retained after applying the pca.
    encoder: category_encoders.utils.BaseEncoder or None
        The encoder that should be applied on the original 'dataset' feature. If 'none' is passed no
        encoder is applied and the feature is just dropped.
    """

    def __init__(
        self,
        nan_ratio_feature_drop_threshold=0.4,
        imputer=SimpleImputer(strategy='mean'),
        scaler=StandardScaler(),
        expected_pca_variance=1,
        encoder=TargetEncoder()
    ):
        self.nan_ratio_feature_drop_threshold = nan_ratio_feature_drop_threshold
        self.imputer = imputer
        self.scaler = scaler
        self.expected_pca_variance = expected_pca_variance
        self.encoder = encoder

    def fit(self, X, y=None):
        metafeatures = get_metafeatures(pd.Series(X['dataset'].unique()))
        ids = metafeatures.index

        features_to_drop = get_features_to_drop_by_nan_threshold(
            metafeatures,
            self.nan_ratio_feature_drop_threshold
        )
        metafeatures = metafeatures.drop(columns=features_to_drop)

        if len(metafeatures.columns) == 0:
            self.metafeatures = pd.DataFrame(index=ids)
        else:
            metafeatures = pd.DataFrame(self.imputer.fit_transform(metafeatures), index=ids)
            metafeatures = pd.DataFrame(self.scaler.fit_transform(metafeatures), index=ids)
            if self.expected_pca_variance < 1:
                metafeatures = pd.DataFrame(
                    PCA(n_components=self.expected_pca_variance).fit_transform(metafeatures),
                    index=ids
                )
            self.metafeatures = metafeatures.add_prefix('dataset_metafeature_')

        if self.encoder:
            try:
                self.encoder.fit(X['dataset'].astype('category'), y)
            except Exception:
                try:
                    self.encoder.fit(X['dataset'].astype('category'), None)
                except Exception as e:
                    print('Encoder requires the target, but the target has an invalid format.')
                    raise

        return self

    def transform(self, X):
        X_new = pd.merge(X, self.metafeatures, left_on='dataset', right_index=True, sort=False).reindex(X.index)
        X_new = X_new.drop(columns=['dataset'])
        if self.encoder:
            X_new = pd.concat([
                X_new,
                self.encoder.transform(X['dataset'].astype('category'))
            ], axis=1)

        return X_new


class GeneralPurposeEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_encoder=BinaryEncoder(),
        tuning_encoder=BinaryEncoder(),
        scoring_encoder=BinaryEncoder()
    ):
        self.model_encoder = model_encoder
        self.tuning_encoder = tuning_encoder
        self.scoring_encoder = scoring_encoder

    def fit(self, X, y=None):
        try:
            self.model_encoder.fit(X['model'], y)
            self.tuning_encoder.fit(X['tuning'], y)
            self.scoring_encoder.fit(X['scoring'], y)
        except Exception:
            try:
                self.model_encoder.fit(X['model'], None)
                self.tuning_encoder.fit(X['tuning'], None)
                self.scoring_encoder.fit(X['scoring'], None)
            except Exception as e:
                print('Encoder requires the target, but the target has an invalid format.')
                raise
        return self

    def transform(self, X):
        model_encoded = self.model_encoder.transform(X['model'])
        tuning_encoded = self.tuning_encoder.transform(X['tuning'])
        scoring_encoded = self.scoring_encoder.transform(X['scoring'])

        X_new = X.drop(columns=['model', 'tuning', 'scoring'])
        X_new = pd.concat([X_new, model_encoded, tuning_encoded, scoring_encoded], axis=1)

        return X_new


def get_column_names(data):
    if isinstance(data, pd.DataFrame):
        return data.columns
    elif isinstance(data, pd.Series):
        return [data.name]
    else:
        return range(get_number_of_columns(data))


def get_number_of_columns(data):
    number_of_columns = 1
    if len(data.shape) == 2:
        number_of_columns = data.shape[1]
    return number_of_columns


def get_transformed_target_column_names(original_target, transformed_target):
    original_name = get_column_names(original_target)[0]
    number_of_columns = get_number_of_columns(transformed_target)
    if number_of_columns == 1:
        return [original_name]
    else:
        return [original_name + '_' + str(i) for i in range(get_number_of_columns(transformed_target))]


def get_inverse_transformed_target_column_names(transformed_target):
    column_names = get_column_names(transformed_target)
    if len(column_names) == 1:
        return column_names
    else:
        return [name.rsplit('_', 1)[0] for name in column_names]


def set_column_names(data, column_names):
    data_new = data.copy()
    if isinstance(data, pd.Series):
        data_new.name = column_names[0]
    else:
        data_new.columns = column_names
    return data_new


def as_pandas_structure(to_convert, index=None, columns=None):
    name = None if columns is None else columns[0]
    if len(to_convert.shape) == 1:
        return pd.Series(to_convert, index=index, name=name)
    elif len(to_convert.shape) == 2 and to_convert.shape[1] == 1:
        if isinstance(to_convert, pd.DataFrame):
            return pd.Series(to_convert.squeeze(), index=index, name=name)
        else:
            return pd.Series(to_convert.ravel(), index=index, name=name)
    else:
        return pd.DataFrame(to_convert, index=index, columns=columns)


class RankingBinarizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self._max_rank = None

    def fit(self, X, y=None):
        # check whether rankings are valid
        all_integers = np.array_equal(X.values, X.values.astype(int))
        no_negatives = (X >= 0).all().item()

        assert all_integers and no_negatives, "Ranking has invalid format"

        self._max_rank = int(X.max().item())

        return self

    def transform(self, X, y=None):
        X_new = X.copy()
        for i in range(self._max_rank):
            X_new[str(i)] = (X > i).astype(int)
        X_new = X_new.drop(columns=X.columns)
        return X_new

    def inverse_transform(self, X, y=None):
        X_new = X.copy()
        transformed_columns = [str(i) for i in range(self._max_rank)]
        X_new = X_new.sum(axis=1)
        X_new = X_new.drop(columns=transformed_columns)
        return X_new


class GroupwiseTargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        transformer,
        group_by=None
    ):
        if group_by is None:
            group_by = []
        self.transformer = transformer
        self.group_by = group_by
        self._default_group = ''
        self._transformers_by_group = {}

    def _fit_target(self, group, y):
        self._transformers_by_group[group] = copy.deepcopy(self.transformer).fit(pd.DataFrame(y))

    def _transform_target(self, group, y, index):
        return as_pandas_structure(self._transformers_by_group[group].transform(pd.DataFrame(y)), index=index)

    def _inverse_transform_target(self, group, y, index):
        return as_pandas_structure(self._transformers_by_group[group].inverse_transform(pd.DataFrame(y)), index=index)

    def fit(self, X, y=None):
        if y is not None:
            if len(self.group_by) == 0:
                self._fit_target(self._default_group, y)
            else:
                df = pd.concat([X, y], axis=1)
                for group, data in df.groupby(by=self.group_by):
                    self._fit_target(group, data[pd.DataFrame(y).columns])
        return self

    def transform(self, X, y=None):
        if y is not None:
            if len(self.group_by) == 0:
                y_new = self._transform_target(self._default_group, y, X.index)
            else:
                df = pd.concat([X, as_pandas_structure(y, index=X.index)], axis=1)
                y_new = pd.Series()
                for group, data in df.groupby(by=self.group_by):
                    y_new = pd.concat([y_new, self._transform_target(group, data[pd.DataFrame(y).columns], data.index)])
                y_new = y_new.reindex(X.index)
            y = set_column_names(y_new, get_transformed_target_column_names(y, y_new))
        return X, y

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y=None):
        if y is not None:
            if len(self.group_by) == 0:
                y_new = self._inverse_transform_target(self._default_group, y, X.index)
            else:
                df = pd.concat([X, as_pandas_structure(y, index=X.index)], axis=1)
                y_new = pd.Series()
                for group, data in df.groupby(by=self.group_by):
                    y_new = pd.concat([
                        y_new,
                        self._inverse_transform_target(group, data[pd.DataFrame(y).columns], data.index)
                    ])
                y_new = y_new.reindex(X.index)
            y = set_column_names(y_new, get_inverse_transformed_target_column_names(y_new))
        return X, y


class TargetScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=MinMaxScaler(feature_range=(0, 1)), group_by=None):
        if group_by is None:
            group_by = ['dataset', 'model', 'tuning', 'scoring']
        self.scaler = scaler
        self.group_by = group_by
        self.scalers_by_group = {}

    def fit(self, X, y=None):
        if y is not None:
            df = pd.concat([X, y], axis=1)
            for group, data in df.groupby(by=self.group_by):
                scaler = copy.deepcopy(self.scaler)
                group_target = pd.DataFrame(data[y.name])
                self.scalers_by_group[group] = scaler.fit(group_target)
        return self

    def transform(self, X, y=None):
        if y is not None:
            df = pd.concat([X, y], axis=1)
            for index, row in df.iterrows():
                group = tuple(row[self.group_by])
                scaler = self.scalers_by_group[group]
                target = pd.DataFrame({y.name: [row[y.name]]})
                df.loc[index, y.name] = scaler.transform(target)
            y = df[y.name]
        return X, y

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y=None):
        if y is not None:
            df = pd.concat([X, y], axis=1)
            for index, row in df.iterrows():
                group = tuple(row[self.group_by])
                scaler = self.scalers_by_group[group]
                target = pd.DataFrame({y.name: [row[y.name]]})
                df.loc[index, y.name] = scaler.inverse_transform(target)
            y = df[y.name]
        return X, y


class TargetOneHotTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.onehot = OneHotEncoder()

    def fit(self, X, y=None):
        if y is not None:
            self.onehot.fit(y.astype('category'))
        return self

    def transform(self, X, y=None):
        if y is not None:
            y = self.onehot.transform(y.astype('category'))
        return X, y

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y=None):
        if y is not None:
            y = self.onehot.inverse_transform(y.astype('category'))
        return X, y
