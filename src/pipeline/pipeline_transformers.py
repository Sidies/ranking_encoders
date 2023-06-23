import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from src.features.embeddings import GraphEmbedding
from src import configuration as config
from sklearn.cluster import KMeans

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
                #print(tabulate(df.columns, headers=df.columns, tablefmt='simple'))
                print(df.columns)
            else:
                df = pd.DataFrame(X)
                #print(tabulate(df.columns, headers=df.columns, tablefmt='simple'))
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
    def __init__(self, graph, **kwargs):
        self.graph = graph
        self.kwargs = kwargs
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        graph_embedding = GraphEmbedding(self.graph)
        model = graph_embedding.node2vec(**self.kwargs)
        
        # Get the embeddings.
        n2v_embeddings = {node: model.wv.get_vector(node) for node in model.wv.index_to_key}
        
        X['node2vec_embedding_dim1'] = 0
        X['node2vec_embedding_dim2'] = 0
        
        print(type(X))
        for i, row in X.iterrows():
            node = row['encoder']
            embedding = n2v_embeddings[node]
            X.at[i, 'node2vec_embedding_dim1'] = embedding[0]
            X.at[i, 'node2vec_embedding_dim2'] = embedding[1]
        return X
    

class Node2VecGraphEmbeddingWithKMeans(BaseEstimator, TransformerMixin):
    def __init__(self, graph, **kwargs):
        self.graph = graph
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        graph_embedding = GraphEmbedding(self.graph)
        model = graph_embedding.node2vec(**self.kwargs)
        
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
        return X
    
    
class PoincareEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, graph, **kwargs):
        self.graph = graph
        self.kwargs = kwargs
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        graph_embedding = GraphEmbedding(self.graph)
        model = graph_embedding.poincare(**self.kwargs)
        
        # Get the embeddings.
        poincare_embeddings = {node: model.kv.get_vector(node) for node in model.kv.index_to_key}
        
        X['poincare_embedding_dim1'] = 0
        X['poincare_embedding_dim2'] = 0
        
        print(type(X))
        for i, row in X.iterrows():
            node = row['encoder']
            embedding = poincare_embeddings[node]
            X.at[i, 'poincare_embedding_dim1'] = embedding[0]
            X.at[i, 'poincare_embedding_dim2'] = embedding[1]
        return X