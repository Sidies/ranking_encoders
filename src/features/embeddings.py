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
        node2vec = Node2Vec(self.graph, **kwargs)
        model = node2vec.fit()
        return model

    def poincare(self, epochs=100, **kwargs):
        # Convert the graph into a list of edges.
        edges = list(self.graph.edges())

        # Train the Poincaré model.
        model = PoincareModel(edges, size=2, negative=2)
        model.train(epochs=epochs)
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