import pandas as pd
import numpy as np

from category_encoders.binary import BinaryEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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
            self.encoder.fit(X['encoder'], y)

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
            self.encoder.fit(X['dataset'].astype('category'), y)

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
        self.model_encoder.fit(X['model'], y)
        self.tuning_encoder.fit(X['tuning'], y)
        self.scoring_encoder.fit(X['scoring'], y)
        return self

    def transform(self, X):
        model_encoded = self.model_encoder.transform(X['model'])
        tuning_encoded = self.tuning_encoder.transform(X['tuning'])
        scoring_encoded = self.scoring_encoder.transform(X['scoring'])

        X_new = X.drop(columns=['model', 'tuning', 'scoring'])
        X_new = pd.concat([X_new, model_encoded, tuning_encoded, scoring_encoded], axis=1)

        return X_new
