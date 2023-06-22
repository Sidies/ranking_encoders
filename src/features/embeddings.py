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

    def poincare(self, eochs=any, batch_size=50, **kwargs):
        model = PoincareModel(self.graph, **kwargs)
        model.train(epochs=eochs, batch_size=batch_size)
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