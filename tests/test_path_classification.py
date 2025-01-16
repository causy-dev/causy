from causy.causal_discovery.constraint.algorithms import PC
from causy.edge_types import DirectedEdge
from causy.graph import GraphBaseAccessMixin, GraphManager
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
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
        self.assertNotIn(
            [node1, node2],
            [l for l in graph.all_paths_on_underlying_undirected_graph(node1, node3)],
        )
        self.assertNotIn(
            [node2, node3],
            [l for l in graph.all_paths_on_underlying_undirected_graph(node1, node3)],
        )
        self.assertNotIn(
            [node1, node3],
            [l for l in graph.all_paths_on_underlying_undirected_graph(node1, node3)],
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
        self.assertNotIn(
            [node1, node2],
            [l for l in graph.all_paths_on_underlying_undirected_graph(node1, node3)],
        )
        self.assertNotIn(
            [node2, node3],
            [l for l in graph.all_paths_on_underlying_undirected_graph(node1, node3)],
        )
        self.assertNotIn(
            [node1, node3],
            [l for l in graph.all_paths_on_underlying_undirected_graph(node1, node3)],
        )

    def test_all_paths_on_underlying_undirected_graph_collider_path(self):
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
        graph.add_directed_edge(node1, node2, {"test": "test"})
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
        self.assertNotIn(
            [node1, node2],
            [l for l in graph.all_paths_on_underlying_undirected_graph(node1, node3)],
        )
        self.assertNotIn(
            [node2, node3],
            [l for l in graph.all_paths_on_underlying_undirected_graph(node1, node3)],
        )
        self.assertNotIn(
            [node1, node3],
            [l for l in graph.all_paths_on_underlying_undirected_graph(node1, node3)],
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
        self.assertNotIn(
            [node1, node2],
            [l for l in graph.all_paths_on_underlying_undirected_graph(node1, node3)],
        )
        self.assertNotIn(
            [node2, node3],
            [l for l in graph.all_paths_on_underlying_undirected_graph(node1, node3)],
        )

    def test_are_nodes_d_separated_open_path_mediated(self):
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
        self.assertFalse(graph.are_nodes_d_separated(node1, node3, []))

    def test_are_nodes_d_separated_blocked_path_mediated(self):
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
        (self.assertTrue(graph.are_nodes_d_separated(node1, node3, [node2])))

    def test_are_nodes_d_separated_blocked_path_mediated(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 1000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        self.assertEqual(
            True,
            tst.graph.are_nodes_d_separated(
                tst.graph.node_by_id("X"),
                tst.graph.node_by_id("Z"),
                [tst.graph.node_by_id("Y")],
            ),
        )

    def test_are_nodes_d_separated_open_path_cpdag(self):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        node3 = graph.add_node("test3", [1, 2, 3])
        # TODO: check add edge again to see if this is the correct way to test cpdags with undirected edges
        graph.add_edge(node1, node2, {"test": "test"})
        graph.add_edge(node3, node2, {"test": "test"})
        (self.assertTrue(graph.are_nodes_d_separated(node1, node3, [node2])))

    def test_are_nodes_d_separated_open_path_confounder(self):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        node3 = graph.add_node("test3", [1, 2, 3])
        graph.add_directed_edge(node2, node1, {"test": "test"})
        graph.add_directed_edge(node2, node3, {"test": "test"})
        self.assertFalse(graph.are_nodes_d_separated(node1, node3, []))

    def test_are_nodes_d_separated_by_conditioning_on_noncollider(self):
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
        self.assertTrue(graph.are_nodes_d_separated(node1, node3, [node2]))

    def test_are_nodes_d_separated_by_a_collider(self):
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
        graph.add_directed_edge(node3, node2, {"test": "test"})
        self.assertTrue(graph.are_nodes_d_separated(node1, node3, []))

    def test_are_nodes_d_separated_open_by_conditioning_on_collider(self):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3], "test1")
        node2 = graph.add_node("test2", [1, 2, 3], "test2")
        node3 = graph.add_node("test3", [1, 2, 3], "test3")
        graph.add_directed_edge(node1, node2, {"test": "test"})
        graph.add_directed_edge(node3, node2, {"test": "test"})
        self.assertFalse(graph.are_nodes_d_separated(node1, node3, [node2]))

    def test_are_nodes_d_separated_two_paths_one_open_one_blocked(self):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3], "test1")
        node2 = graph.add_node("test2", [1, 2, 3], "test2")
        node3 = graph.add_node("test3", [1, 2, 3], "test3")
        graph.add_directed_edge(node1, node2, {"test": "test"})
        graph.add_directed_edge(node3, node2, {"test": "test"})
        graph.add_directed_edge(node1, node3, {"test": "test"})
        self.assertFalse(graph.are_nodes_d_separated(node1, node3, []))

    def test_are_nodes_d_separated_no_path_empty_conditioning_set(self):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3], "test1")
        node2 = graph.add_node("test2", [1, 2, 3], "test2")
        node3 = graph.add_node("test3", [1, 2, 3], "test3")
        graph.add_directed_edge(node1, node2, {"test": "test"})
        self.assertTrue(graph.are_nodes_d_separated(node1, node3, []))

    def test_are_nodes_d_separated_no_path_with_nonempty_conditioning_set(self):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3], "test1")
        node2 = graph.add_node("test2", [1, 2, 3], "test2")
        node3 = graph.add_node("test3", [1, 2, 3], "test3")
        graph.add_directed_edge(node1, node2, {"test": "test"})
        self.assertTrue(graph.are_nodes_d_separated(node1, node3, [node2]))

    def test_are_nodes_d_separated_connected_by_conditioning_on_collider_with_more_descendants(
        self,
    ):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3], "test1")
        node2 = graph.add_node("test2", [1, 2, 3], "test2")
        node3 = graph.add_node("test3", [1, 2, 3], "test3")
        node4 = graph.add_node("test4", [1, 2, 3], "test4")
        node5 = graph.add_node("test5", [1, 2, 3], "test5")
        graph.add_directed_edge(node1, node2, {"test": "test"})
        graph.add_directed_edge(node3, node2, {"test": "test"})
        graph.add_directed_edge(node2, node4, {"test": "test"})
        graph.add_directed_edge(node2, node5, {"test": "test"})
        self.assertFalse(graph.are_nodes_d_separated(node1, node3, [node2]))

    def test_are_nodes_d_separated_connected_by_conditioning_on_descendant_of_collider(
        self,
    ):
        new_graph_manager = GraphManager
        new_graph_manager.__bases__ = (
            GraphBaseAccessMixin,
            DirectedEdge.GraphAccessMixin,
        )
        graph = new_graph_manager()
        node1 = graph.add_node("test1", [1, 2, 3], "test1")
        node2 = graph.add_node("test2", [1, 2, 3], "test2")
        node3 = graph.add_node("test3", [1, 2, 3], "test3")
        node4 = graph.add_node("test4", [1, 2, 3], "test4")
        node5 = graph.add_node("test5", [1, 2, 3], "test5")
        graph.add_directed_edge(node1, node2, {"test": "test"})
        graph.add_directed_edge(node3, node2, {"test": "test"})
        graph.add_directed_edge(node2, node4, {"test": "test"})
        graph.add_directed_edge(node2, node5, {"test": "test"})
        self.assertFalse(graph.are_nodes_d_separated(node1, node3, [node4]))
