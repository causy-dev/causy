import csv
import unittest
import numpy as np
import torch

from causy.algorithms import PC, ParallelPC
from causy.graph_utils import retrieve_edges
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference


# TODO: generate larger toy model to test quadruple orientation rules.
# TODO: seedings are not working yet (failing every 20th time or so, should always be equal for equal data), fix that.


def set_random_seed(seed):
    # Ensure reproducability across operating systems
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


class PCTestTestCase(unittest.TestCase):
    def test_toy_model_minimal_example(self):
        set_random_seed(1)
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("Z"), NodeReference("X"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("V"), NodeReference("Z"), 1),
            ]
        )

        model.random_fn = lambda: torch.normal(0, 1, (1, 1))
        sample_size = 10000
        test_data = model._generate_shaped_data(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        retrieve_edges(tst.graph)

        node_mapping = {}
        for key, node in tst.graph.nodes.items():
            node_mapping[node.name] = key

        self.assertAlmostEqual(
            tst.graph.edges[node_mapping["Z"]][node_mapping["X"]].metadata[
                "direct_effect"
            ],
            5.0,
            1,
        )

        self.assertAlmostEqual(
            tst.graph.edges[node_mapping["V"]][node_mapping["Z"]].metadata[
                "direct_effect"
            ],
            1.0,
            1,
        )
        self.assertAlmostEqual(
            tst.graph.edges[node_mapping["W"]][node_mapping["Z"]].metadata[
                "direct_effect"
            ],
            1.0,
            1,
        )

        self.assertTrue(
            tst.graph.only_directed_edge_exists(
                tst.graph.nodes[node_mapping["V"]], tst.graph.nodes[node_mapping["Z"]]
            )
        )
        self.assertTrue(
            tst.graph.only_directed_edge_exists(
                tst.graph.nodes[node_mapping["W"]], tst.graph.nodes[node_mapping["Z"]]
            )
        )
        self.assertTrue(
            tst.graph.only_directed_edge_exists(
                tst.graph.nodes[node_mapping["Z"]], tst.graph.nodes[node_mapping["X"]]
            )
        )
        self.assertTrue(
            tst.graph.only_directed_edge_exists(
                tst.graph.nodes[node_mapping["Z"]], tst.graph.nodes[node_mapping["Y"]]
            )
        )

    def test_second_toy_model_example(self):
        set_random_seed(1)
        c, d, e, f, g = 2, 3, 4, 5, 6
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("A"), NodeReference("C"), 1),
                SampleEdge(NodeReference("B"), NodeReference("C"), c),
                SampleEdge(NodeReference("A"), NodeReference("D"), d),
                SampleEdge(NodeReference("B"), NodeReference("D"), 1),
                SampleEdge(NodeReference("C"), NodeReference("D"), 1),
                SampleEdge(NodeReference("B"), NodeReference("E"), e),
                SampleEdge(NodeReference("E"), NodeReference("F"), f),
                SampleEdge(NodeReference("B"), NodeReference("F"), g),
                SampleEdge(NodeReference("C"), NodeReference("F"), 1),
                SampleEdge(NodeReference("D"), NodeReference("F"), 1),
            ],
        )

        model.random_fn = lambda: torch.normal(0, 1, (1, 1))
        sample_size = 10000
        test_data = model._generate_shaped_data(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        retrieve_edges(tst.graph)

        node_mapping = {}
        for key, node in tst.graph.nodes.items():
            node_mapping[node.name] = key

        self.assertFalse(
            tst.graph.edge_exists(
                tst.graph.nodes[node_mapping["A"]], tst.graph.nodes[node_mapping["B"]]
            )
        )

        self.assertFalse(
            tst.graph.edge_exists(
                tst.graph.nodes[node_mapping["B"]], tst.graph.nodes[node_mapping["A"]]
            )
        )

        self.assertTrue(
            tst.graph.directed_edge_exists(
                tst.graph.nodes[node_mapping["A"]], tst.graph.nodes[node_mapping["C"]]
            )
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(
                tst.graph.nodes[node_mapping["B"]], tst.graph.nodes[node_mapping["C"]]
            )
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(
                tst.graph.nodes[node_mapping["A"]], tst.graph.nodes[node_mapping["D"]]
            )
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(
                tst.graph.nodes[node_mapping["B"]], tst.graph.nodes[node_mapping["D"]]
            )
        )

        self.assertTrue(
            tst.graph.directed_edge_exists(
                tst.graph.nodes[node_mapping["B"]], tst.graph.nodes[node_mapping["E"]]
            )
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(
                tst.graph.nodes[node_mapping["E"]], tst.graph.nodes[node_mapping["F"]]
            )
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(
                tst.graph.nodes[node_mapping["B"]], tst.graph.nodes[node_mapping["F"]]
            )
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(
                tst.graph.nodes[node_mapping["C"]], tst.graph.nodes[node_mapping["F"]]
            )
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(
                tst.graph.nodes[node_mapping["D"]], tst.graph.nodes[node_mapping["F"]]
            )
        )
        self.assertFalse(
            tst.graph.edge_exists(
                tst.graph.nodes[node_mapping["A"]], tst.graph.nodes[node_mapping["E"]]
            )
        )

        self.assertFalse(
            tst.graph.edge_exists(
                tst.graph.nodes[node_mapping["E"]], tst.graph.nodes[node_mapping["D"]]
            )
        )

        self.assertFalse(
            tst.graph.edge_exists(
                tst.graph.nodes[node_mapping["E"]], tst.graph.nodes[node_mapping["C"]]
            )
        )


if __name__ == "__main__":
    unittest.main()
