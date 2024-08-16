from causy.edge_types import DirectedEdge
from causy.graph import GraphBaseAccessMixin, GraphManager
from tests.utils import CausyTestCase


class GraphTestCase(CausyTestCase):
    def test_all_paths_on_underlying_undirected_graph(self):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        node3 = graph.add_node("test3", [1, 2, 3])
        graph.add_directed_edge(node1, node2, {"test": "test"})
        graph.add_directed_edge(node2, node3, {"test": "test"})
        self.assertIn(
            [node1, node2, node3],
            [
                x
                for x in [
                    l
                    for l in graph.all_paths_on_underlying_undirected_graph(
                        node1, node3
                    )
                ]
            ],
        )

    def test_all_paths_on_underlying_undirected_graph_2(self):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        node3 = graph.add_node("test3", [1, 2, 3])
        graph.add_directed_edge(node3, node2, {"test": "test"})
        graph.add_directed_edge(node2, node1, {"test": "test"})
        self.assertIn(
            [node1, node2, node3],
            [
                x
                for x in [
                    l
                    for l in graph.all_paths_on_underlying_undirected_graph(
                        node1, node3
                    )
                ]
            ],
        )

    def test_all_paths_on_underlying_undirected_graph_several_paths(self):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        node3 = graph.add_node("test3", [1, 2, 3])
        graph.add_directed_edge(node1, node2, {"test": "test"})
        graph.add_directed_edge(node2, node3, {"test": "test"})
        graph.add_directed_edge(node1, node3, {"test": "test"})
        self.assertIn(
            [node1, node2, node3],
            [
                x
                for x in [
                    l
                    for l in graph.all_paths_on_underlying_undirected_graph(
                        node1, node3
                    )
                ]
            ],
        )
        self.assertIn(
            [node1, node3],
            [
                x
                for x in [
                    l
                    for l in graph.all_paths_on_underlying_undirected_graph(
                        node1, node3
                    )
                ]
            ],
        )
