from node2vec import Node2Vec
from gensim.models.poincare import PoincareModel
from pyvis.network import Network
from src import configuration as config
from src.features.encoder_utils import load_graph

class GraphEmbedding:
    def __init__(self, graph):
        """
        Initialize the class with the graph to be embedded.

        Args:
            graph (Graph): The graph to be embedded
        """
        self.graph = graph

    def node2vec(self, **kwargs):
        """
        Train the node2vec model on the graph.
        example parameters: walk_length=20, num_walks=1000, workers=1
        """        
        node2vec = Node2Vec(self.graph, **kwargs)
        model = node2vec.fit()
        return model

    def poincare(self, epochs=100, batch_size=10, **kwargs):
        """
        Train the Poincaré model on the graph.
        example parameters: size=2, negative=2
        """
        # Convert the graph into a list of edges.
        edges = list(self.graph.edges())

        # Train the Poincaré model.
        model = PoincareModel(edges, **kwargs)
        model.train(epochs=epochs, batch_size=batch_size)
        return model

    def visualize(self, model):
        net = Network(notebook=True)
        net.from_nx(self.graph)
        for node in self.graph.nodes():
            net.add_node(node, label=node)
        for edge in self.graph.edges():
            net.add_edge(edge[0], edge[1])
        net.show_buttons(filter_=['physics'])
        net.show("graph.html")