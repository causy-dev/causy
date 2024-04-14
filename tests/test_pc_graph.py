import csv
import torch

from causy.algorithms import PC, ParallelPC
from causy.algorithms.pc import PCStable
from causy.graph_utils import retrieve_edges
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference

from tests.utils import CausyTestCase

# TODO: generate larger toy model to test quadruple orientation rules.
# TODO: seedings are not working yet (failing every 20th time or so, should always be equal for equal data), fix that.


class PCTestTestCase(CausyTestCase):
    def test_with_rki_data(self):
        with open("./tests/fixtures/rki-data.csv") as f:
            data = csv.DictReader(f)
            test_data = []
            for row in data:
                for k in row.keys():
                    if row[k] == "":
                        row[k] = 0.0
                    row[k] = float(row[k])
                test_data.append(row)
        self.assertEqual(len(test_data), 401)
        self.assertEqual(len(test_data[0]), 7)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        retrieve_edges(tst.graph)

    def test_toy_model_minimal_example(self):
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("Z"), NodeReference("X"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("V"), NodeReference("Z"), 1),
            ],
            random=lambda: torch.normal(0, 1, (1, 1)),
        )
        sample_size = 10000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

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
            random=lambda: torch.normal(0, 1, (1, 1)),
        )

        sample_size = 10000
        test_data, sample_graph = model.generate(sample_size)

        tst = PCStable()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        # TODO(sofia): this test is failing - maybe because the graph is not discovered correctly?
        self.assertGraphStructureIsEqual(tst.graph, sample_graph)

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

    # test structure learning
    def test_toy_model_structure(self):
        """
        Test conditional independence of pairs given one variable works.
        """
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Y"), 2),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 3),
                SampleEdge(NodeReference("X"), NodeReference("W"), 4),
            ],
            random=lambda: torch.normal(0, 1, (1, 1)),
        )
        sample_size = 10000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_toy_model_structure_2(self):
        """
        Another test if conditional independence of pairs given one variable works.
        """
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 6),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 3),
            ],
            random=lambda: torch.normal(0, 1, (1, 1)),
        )
        sample_size = 10000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_toy_model_structure_3(self):
        """
        Test conditional independence of ordered pairs given pairs of other variables works.
        """
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 3),
                SampleEdge(NodeReference("X"), NodeReference("F"), 4),
                SampleEdge(NodeReference("F"), NodeReference("W"), 7),
            ],
            random=lambda: torch.normal(0, 1, (1, 1)),
        )
        sample_size = 10000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_toy_model_structure_4(self):
        """
        Test conditional independence of ordered pairs given triples of other variables works.
        """
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 3),
                SampleEdge(NodeReference("X"), NodeReference("F"), 4),
                SampleEdge(NodeReference("F"), NodeReference("W"), 7),
                SampleEdge(NodeReference("X"), NodeReference("D"), 4),
                SampleEdge(NodeReference("D"), NodeReference("W"), 7),
            ],
            random=lambda: torch.normal(0, 1, (1, 1)),
        )
        sample_size = 10000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_toy_model_structure_4(self):
        """
        Test conditional independence of ordered pairs given quadruples of other variables works.
        """
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 3),
                SampleEdge(NodeReference("X"), NodeReference("F"), 4),
                SampleEdge(NodeReference("F"), NodeReference("W"), 7),
                SampleEdge(NodeReference("X"), NodeReference("D"), 4),
                SampleEdge(NodeReference("D"), NodeReference("W"), 7),
                SampleEdge(NodeReference("X"), NodeReference("Q"), 4),
                SampleEdge(NodeReference("Q"), NodeReference("W"), 7),
            ],
            random=lambda: torch.normal(0, 1, (1, 1)),
        )
        sample_size = 10000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

    # test causal orientation rules

    def test_toy_model_orientation_unshielded_triple_collider(self):
        """
        Test if orientation of edges work: Minimal example with empty separation set (collider case, unshielded triples).
        """
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Z"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
            ],
            random=lambda: torch.normal(0, 1, (1, 1)),
        )
        sample_size = 10000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)
        self.assertTrue(
            tst.graph.directed_edge_exists(tst.graph.nodes["X"], tst.graph.nodes["Z"])
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(tst.graph.nodes["Y"], tst.graph.nodes["Z"])
        )
        self.assertFalse(
            tst.graph.directed_edge_exists(tst.graph.nodes["Z"], tst.graph.nodes["Y"])
        )
        self.assertFalse(
            tst.graph.directed_edge_exists(tst.graph.nodes["Z"], tst.graph.nodes["X"])
        )

    def test_toy_model_orientation_unshielded_triple_non_collider(self):
        """
        Test if orientation of edges work: unshielded triple, further non-collider test.
        """
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Z"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
                SampleEdge(NodeReference("Z"), NodeReference("D"), 4),
            ],
            random=lambda: torch.normal(0, 1, (1, 1)),
        )
        sample_size = 10000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        self.assertTrue(
            tst.graph.directed_edge_exists(tst.graph.nodes["X"], tst.graph.nodes["Z"])
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(tst.graph.nodes["Y"], tst.graph.nodes["Z"])
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(tst.graph.nodes["Z"], tst.graph.nodes["D"])
        )
        self.assertFalse(
            tst.graph.directed_edge_exists(tst.graph.nodes["Z"], tst.graph.nodes["Y"])
        )
        self.assertFalse(
            tst.graph.directed_edge_exists(tst.graph.nodes["Z"], tst.graph.nodes["X"])
        )
        self.assertFalse(
            tst.graph.directed_edge_exists(tst.graph.nodes["D"], tst.graph.nodes["Z"])
        )

    def test_toy_model_orientation_unshielded_triple_non_collider2(self):
        """
        Test if orientation of edges work: unshielded triple, further non-collider tests.
        """
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Z"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
                SampleEdge(NodeReference("Z"), NodeReference("D"), 4),
                SampleEdge(NodeReference("Z"), NodeReference("Q"), 4),
            ],
            random=lambda: torch.normal(0, 1, (1, 1)),
        )
        sample_size = 10000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        self.assertTrue(
            tst.graph.directed_edge_exists(tst.graph.nodes["X"], tst.graph.nodes["Z"])
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(tst.graph.nodes["Y"], tst.graph.nodes["Z"])
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(tst.graph.nodes["Z"], tst.graph.nodes["D"])
        )
        self.assertTrue(
            tst.graph.directed_edge_exists(tst.graph.nodes["Z"], tst.graph.nodes["Q"])
        )
        self.assertFalse(
            tst.graph.directed_edge_exists(tst.graph.nodes["Z"], tst.graph.nodes["Y"])
        )
        self.assertFalse(
            tst.graph.directed_edge_exists(tst.graph.nodes["Z"], tst.graph.nodes["X"])
        )
        self.assertFalse(
            tst.graph.directed_edge_exists(tst.graph.nodes["D"], tst.graph.nodes["Z"])
        )
        self.assertFalse(
            tst.graph.directed_edge_exists(tst.graph.nodes["Q"], tst.graph.nodes["Z"])
        )
