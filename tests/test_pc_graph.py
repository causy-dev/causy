import csv

from causy.causal_discovery.constraint.algorithms import PC
from causy.causal_discovery.constraint.algorithms.pc import (
    PC_ORIENTATION_RULES,
    PC_EDGE_TYPES,
    PC_GRAPH_UI_EXTENSION,
    PC_DEFAULT_THRESHOLD,
)
from causy.causal_discovery.constraint.independence_tests.common import (
    CorrelationCoefficientTest,
    PartialCorrelationTest,
    ExtendedPartialCorrelationTestMatrix,
)
from causy.causal_effect_estimation.multivariate_regression import (
    ComputeDirectEffectsInDAGsMultivariateRegression,
)
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.graph_model import graph_model_factory
from causy.graph_utils import retrieve_edges
from causy.models import Algorithm
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from causy.variables import VariableReference, FloatVariable

from tests.utils import CausyTestCase

# TODO: generate larger toy model to test quadruple orientation rules.
# TODO: seedings are not working yet (failing every 20th time or so, should always be equal for equal data), fix that.


class PCTestTestCase(CausyTestCase):
    SEED = 1

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
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("Z"), NodeReference("X"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("V"), NodeReference("Z"), 1),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 100000
        tst = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(
                        display_name="Calculate Pearson Correlations"
                    ),
                    CorrelationCoefficientTest(
                        threshold=VariableReference(name="threshold"),
                        display_name="Correlation Coefficient Test",
                    ),
                    PartialCorrelationTest(
                        threshold=VariableReference(name="threshold"),
                        display_name="Partial Correlation Test",
                    ),
                    ExtendedPartialCorrelationTestMatrix(
                        threshold=VariableReference(name="threshold"),
                        display_name="Extended Partial Correlation Test Matrix",
                    ),
                    *PC_ORIENTATION_RULES,
                    ComputeDirectEffectsInDAGsMultivariateRegression(
                        display_name="Compute Direct Effects"
                    ),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[PC_GRAPH_UI_EXTENSION],
                name="PC",
                variables=[FloatVariable(name="threshold", value=PC_DEFAULT_THRESHOLD)],
            )
        )()
        test_data, graph = model.generate(sample_size)
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

    # test structure learning
    def test_toy_model_structure(self):
        """
        Test conditional independence of pairs given one variable works.
        """
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Y"), 2),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 3),
                SampleEdge(NodeReference("X"), NodeReference("W"), 4),
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

    def test_toy_model_structure_2(self):
        """
        Another test if conditional independence of pairs given one variable works.
        """
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 6),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 3),
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

    def test_toy_model_structure_3(self):
        """
        Test conditional independence of ordered pairs given pairs of other variables works.
        """
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 3),
                SampleEdge(NodeReference("X"), NodeReference("F"), 4),
                SampleEdge(NodeReference("F"), NodeReference("W"), 7),
            ],
            random=lambda: rdnv(0, 1),
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
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 3),
                SampleEdge(NodeReference("X"), NodeReference("F"), 4),
                SampleEdge(NodeReference("F"), NodeReference("W"), 7),
            ],
            random=lambda: rdnv(0, 1),
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
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Z"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
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
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Z"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
                SampleEdge(NodeReference("Z"), NodeReference("D"), 4),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 100000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        # self.assertGraphStructureIsEqual(tst.graph, graph)

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
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Z"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 2),
                SampleEdge(NodeReference("Z"), NodeReference("D"), 4),
                SampleEdge(NodeReference("Z"), NodeReference("Q"), 4),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 100000
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

    def test_with_minimal_toy_model_not_disjoint_graph(self):
        """
        Test if orientation of edges work: Minimal example with empty separation set (collider case, unshielded triples).
        """
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 100000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        self.assertFalse(
            tst.graph.only_directed_edge_exists(
                tst.graph.nodes["X"], tst.graph.nodes["Y"]
            )
        )
        self.assertFalse(
            tst.graph.only_directed_edge_exists(
                tst.graph.nodes["Y"], tst.graph.nodes["X"]
            )
        )

    def test_with_toy_model_disjoint_graph(self):
        """
        Test if orientation of edges work: disjoint graph.
        """
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 2),
            ],
            random=lambda: rdnv(0, 1),
        )
        sample_size = 100000
        test_data, graph = model.generate(sample_size)

        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        self.assertTrue(
            tst.graph.edge_exists(tst.graph.nodes["X"], tst.graph.nodes["Y"])
        )
        self.assertTrue(
            tst.graph.edge_exists(tst.graph.nodes["Z"], tst.graph.nodes["W"])
        )
        self.assertFalse(
            tst.graph.only_directed_edge_exists(
                tst.graph.nodes["X"], tst.graph.nodes["Y"]
            )
        )
        self.assertFalse(
            tst.graph.only_directed_edge_exists(
                tst.graph.nodes["Y"], tst.graph.nodes["X"]
            )
        )
        self.assertFalse(
            tst.graph.only_directed_edge_exists(
                tst.graph.nodes["Z"], tst.graph.nodes["W"]
            )
        )
        self.assertFalse(
            tst.graph.only_directed_edge_exists(
                tst.graph.nodes["W"], tst.graph.nodes["Z"]
            )
        )
        self.assertFalse(
            tst.graph.edge_exists(tst.graph.nodes["Y"], tst.graph.nodes["Z"])
        )
        self.assertFalse(
            tst.graph.edge_exists(tst.graph.nodes["W"], tst.graph.nodes["X"])
        )
