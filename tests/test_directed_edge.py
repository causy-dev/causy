from causy.edge_types import DirectedEdge
from causy.graph import GraphManager, Graph, GraphBaseAccessMixin
from causy.interfaces import BaseGraphInterface
from tests.utils import CausyTestCase


class DirectedEdgeTestCase(CausyTestCase):
    def test_directed_paths_two_nodes(self):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        graph.add_directed_edge(node1, node2, {"test": "test"})
        self.assertEqual(graph.directed_paths(node1, node2), [[(node1, node2)]])

    def test_directed_paths_three_nodes(self):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        node3 = graph.add_node("test2", [1, 2, 3])
        graph.add_directed_edge(node1, node2, {"test": "test"})
        graph.add_directed_edge(node2, node3, {"test": "test"})
        self.assertEqual(
            graph.directed_paths(node1, node3), [[(node1, node2), (node2, node3)]]
        )
