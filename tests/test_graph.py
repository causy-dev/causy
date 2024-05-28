import torch

from causy.graph import GraphManager, GraphError, Node

from tests.utils import CausyTestCase


class GraphTestCase(CausyTestCase):
    def test_add_node(self):
        graph = GraphManager()
        node = graph.add_node("test", [1, 2, 3])
        self.assertEqual(node.name, "test")
        self.assertTrue(
            torch.equal(node.values, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        )
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(len(graph.edges), 0)

    def test_add_node_custom_id(self):
        graph = GraphManager()
        node = graph.add_node("test", [1, 2, 3], id_="custom")
        self.assertEqual(node.name, "test")
        self.assertTrue(
            torch.equal(node.values, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        )
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(len(graph.edges), 0)
        self.assertEqual(node.id, "custom")

    def test_add_node_custom_id_already_exists(self):
        graph = GraphManager()
        graph.add_node("test", [1, 2, 3], id_="custom")
        with self.assertRaises(ValueError):
            graph.add_node("test", [1, 2, 3], id_="custom")

    def test_add_node_with_unsupported_values(self):
        graph = GraphManager()
        with self.assertRaises(ValueError):
            graph.add_node("test", [1, 2, 3, "a"])

    def test_add_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        graph.add_edge(node1, node2, {"test": "test"})
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(
            len(graph.edges), 2
        )  # undirected edge is directed edge in both directions
        self.assertEqual(graph.edge_value(node1, node2), {"test": "test"})
        self.assertEqual(graph.edge_value(node2, node1), {"test": "test"})
        self.assertTrue(graph.edge_exists(node1, node2))

    def test_add_directed_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        graph.add_directed_edge(node1, node2, {"test": "test"})
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edge_value(node1, node2), {"test": "test"})
        self.assertTrue(graph.directed_edge_exists(node1, node2))
        self.assertFalse(graph.directed_edge_exists(node2, node1))

    def test_add_edge_with_non_existing_node(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = Node(name="test2", values=torch.Tensor([1, 2, 3]), id="custom")
        with self.assertRaises(GraphError):
            graph.add_edge(node1, node2, {"test": "test"})

        with self.assertRaises(GraphError):
            graph.add_edge(node2, node1, {"test": "test"})

    def test_add_self_loop(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        with self.assertRaises(GraphError):
            graph.add_edge(node1, node1, {"test": "test"})

    def test_remove_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        graph.add_edge(node1, node2, {"test": "test"})
        graph.remove_edge(node1, node2)
        self.assertFalse(graph.edge_exists(node1, node2))
        self.assertFalse(graph.edge_exists(node2, node1))

    def test_remove_edge_with_non_existing_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        self.assertIsNone(graph.remove_edge(node1, node2))

    def test_remove_edge_with_non_existing_node(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = Node(name="test2", values=torch.Tensor([1, 2, 3]), id="custom")
        with self.assertRaises(GraphError):
            graph.remove_edge(node1, node2)

        with self.assertRaises(GraphError):
            graph.remove_edge(node2, node1)

    def test_remove_directed_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        graph.add_edge(node1, node2, {"test": "test"})
        graph.remove_directed_edge(node1, node2)
        self.assertFalse(graph.directed_edge_exists(node1, node2))
        self.assertTrue(graph.directed_edge_exists(node2, node1))

    def test_remove_directed_edge_with_non_existing_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        self.assertIsNone(graph.remove_directed_edge(node1, node2))

        graph.add_edge(node1, node2, {"test": "test"})
        graph.remove_directed_edge(node1, node2)
        self.assertIsNone(graph.remove_directed_edge(node1, node2))

    def test_remove_directed_edge_with_non_existing_node(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = Node(name="test2", values=torch.Tensor([1, 2, 3]), id="custom")
        with self.assertRaises(GraphError):
            graph.remove_directed_edge(node1, node2)

        with self.assertRaises(GraphError):
            graph.remove_directed_edge(node2, node1)

    def test_update_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        graph.add_edge(node1, node2, {"test": "test"})
        graph.update_edge(node1, node2, {"test": "test2"})
        self.assertEqual(graph.edge_value(node1, node2), {"test": "test2"})
        self.assertEqual(graph.edge_value(node2, node1), {"test": "test2"})

    def test_update_edge_with_non_existing_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        with self.assertRaises(GraphError):
            graph.update_edge(node2, node1, {"test": "test"})

    def test_update_edge_with_non_existing_node(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = Node(name="test2", values=torch.Tensor([1, 2, 3]), id="custom")
        with self.assertRaises(GraphError):
            graph.update_edge(node1, node2, {"test": "test"})

        with self.assertRaises(GraphError):
            graph.update_edge(node2, node1, {"test": "test"})

    def test_update_directed_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        graph.add_edge(node1, node2, {"test": "test"})
        graph.update_directed_edge(node1, node2, {"test": "test2"})
        self.assertEqual(graph.edge_value(node1, node2), {"test": "test2"})
        self.assertEqual(graph.edge_value(node2, node1), {"test": "test"})

    def test_update_directed_edge_with_non_existing_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])

        with self.assertRaises(GraphError):
            graph.update_directed_edge(node1, node2, {"test": "test"})

        graph.add_edge(node1, node2, {"test": "test"})
        graph.remove_directed_edge(node2, node1)

        with self.assertRaises(GraphError):
            graph.update_directed_edge(node2, node1, {"test": "test"})

    def test_update_directed_edge_with_non_existing_node(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = Node(name="test2", values=torch.Tensor([1, 2, 3]), id="custom")

        with self.assertRaises(GraphError):
            graph.update_directed_edge(node1, node2, {"test": "test"})

        with self.assertRaises(GraphError):
            graph.update_directed_edge(node2, node1, {"test": "test"})

    def test_edge_value_with_non_existing_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = Node(name="test2", values=torch.Tensor([1, 2, 3]), id="custom")

        self.assertIsNone(graph.edge_value(node1, node2))
        self.assertIsNone(graph.edge_value(node2, node1))

    def test_edge_exists(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node(name="test2", values=[1, 2, 3])
        graph.add_edge(node1, node2, {"test": "test"})
        self.assertTrue(graph.edge_exists(node1, node2))
        self.assertTrue(graph.edge_exists(node2, node1))

    def test_edge_exists_with_non_existing_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node(name="test2", values=[1, 2, 3])
        self.assertFalse(graph.edge_exists(node1, node2))
        self.assertFalse(graph.edge_exists(node2, node1))

    def test_undirected_edge_exists(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node(name="test2", values=[1, 2, 3])
        graph.add_edge(node1, node2, {"test": "test"})
        self.assertTrue(graph.undirected_edge_exists(node1, node2))
        self.assertTrue(graph.undirected_edge_exists(node2, node1))

    def test_undirected_edge_exists_with_non_existing_edge(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node(name="test2", values=[1, 2, 3])
        self.assertFalse(graph.undirected_edge_exists(node1, node2))
        self.assertFalse(graph.undirected_edge_exists(node2, node1))

        graph.add_edge(node1, node2, {"test": "test"})
        graph.remove_directed_edge(node1, node2)

        self.assertFalse(graph.undirected_edge_exists(node1, node2))
        self.assertFalse(graph.undirected_edge_exists(node2, node1))

    def test_parents_of_node_two_nodes(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        graph.add_directed_edge(node1, node2, {"test": "test"})
        self.assertEqual(graph.parents_of_node(node2), [node1])

    def test_parents_of_node_three_nodes(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        node3 = graph.add_node("test2", [1, 2, 3])
        graph.add_directed_edge(node1, node2, {"test": "test"})
        graph.add_directed_edge(node3, node2, {"test": "test"})
        result = graph.parents_of_node(node2)
        self.assertIn(node1, result)
        self.assertIn(node3, result)

    def test_directed_paths_two_nodes(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        graph.add_directed_edge(node1, node2, {"test": "test"})
        self.assertEqual(graph.directed_paths(node1, node2), [[(node1, node2)]])

    def test_directed_paths_three_nodes(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        node3 = graph.add_node("test2", [1, 2, 3])
        graph.add_directed_edge(node1, node2, {"test": "test"})
        graph.add_directed_edge(node2, node3, {"test": "test"})
        self.assertEqual(
            graph.directed_paths(node1, node3), [[(node1, node2), (node2, node3)]]
        )

    def test_inducing_path_exists_basic(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        node3 = graph.add_node("test3", [1, 2, 3])
        graph.add_directed_edge(node1, node2, {"test": "test"})
        graph.add_directed_edge(node2, node3, {"test": "test"})
        self.assertTrue(graph.inducing_path_exists(node1, node2))
        self.assertTrue(graph.inducing_path_exists(node2, node3))
        self.assertFalse(graph.inducing_path_exists(node1, node3))

    def test_inducing_path_exists(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        node3 = graph.add_node("test3", [1, 2, 3])
        graph.add_bidirected_edge(node1, node2, {"test": "test"})
        graph.add_directed_edge(node2, node3, {"test": "test"})
        self.assertTrue(graph.inducing_path_exists(node1, node2))
        self.assertTrue(graph.inducing_path_exists(node2, node3))
        self.assertTrue(graph.inducing_path_exists(node1, node3))

    def test_inducing_path_exists(self):
        graph = GraphManager()
        node1 = graph.add_node("test1", [1, 2, 3])
        node2 = graph.add_node("test2", [1, 2, 3])
        node3 = graph.add_node("test3", [1, 2, 3])
        node4 = graph.add_node("test4", [1, 2, 3])
        graph.add_bidirected_edge(node1, node2, {"test": "test"})
        graph.add_bidirected_edge(node2, node3, {"test": "test"})
        graph.add_bidirected_edge(node3, node4, {"test": "test"})
        graph.add_directed_edge(node2, node4, {"test": "test"})
        graph.add_directed_edge(node3, node1, {"test": "test"})
        self.assertTrue(graph.inducing_path_exists(node1, node4))
        self.assertTrue(graph.inducing_path_exists(node2, node3))
