from causy.causal_discovery.constraint.algorithms.pc import (
    PC_EDGE_TYPES,
    PC,
    PC_ORIENTATION_RULES,
    PC_GRAPH_UI_EXTENSION,
    PC_DEFAULT_THRESHOLD, PCClassic,
)
from causy.causal_effect_estimation.multivariate_regression import (
    ComputeDirectEffectsInDAGsMultivariateRegression,
)
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.generators import PairsWithNeighboursGenerator
from causy.graph_model import graph_model_factory
from causy.causal_discovery.constraint.independence_tests.common import (
    CorrelationCoefficientTest,
    PartialCorrelationTest,
    ExtendedPartialCorrelationTestMatrix,
)
from causy.interfaces import AS_MANY_AS_FIELDS
from causy.models import Algorithm, ComparisonSettings
from causy.causal_discovery.constraint.orientation_rules.pc import ColliderTest
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from causy.variables import VariableReference, FloatVariable

from tests.utils import CausyTestCase, load_fixture_graph


class PCTestTestCase(CausyTestCase):
    SEED = 1

    def _sample_generator(self):
        rdnv = self.seeded_random.normalvariate
        return IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Y"), 2),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 3),
                SampleEdge(NodeReference("X"), NodeReference("W"), 4),
            ],
            random=lambda: rdnv(0, 1),
        )

    def test_pc_number_of_all_proposed_actions_two_nodes(self):
        """
        test if the number of all proposed actions is correct
        """
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(1000)
        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        pc_results = tst.execute_pipeline_steps()
        self.assertEqual(len(pc_results[0].all_proposed_actions), 1)
        self.assertEqual(len(pc_results[1].all_proposed_actions), 1)
        self.assertEqual(len(pc_results[2].all_proposed_actions), 0)

    def test_pc_number_of_actions_two_nodes(self):
        """
        test if the number of all actions is correct
        """
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(1000)
        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        pc_results = tst.execute_pipeline_steps()
        self.assertEqual(len(pc_results[0].actions), 1)
        self.assertEqual(len(pc_results[1].actions), 0)
        self.assertEqual(len(pc_results[2].actions), 0)

    def test_pc_number_of_all_proposed_actions_three_nodes(self):
        """
        test if the number of all proposed actions is correct
        """
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10)
        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        pc_results = tst.execute_pipeline_steps()
        self.assertEqual(len(pc_results[0].all_proposed_actions), 3)
        self.assertEqual(len(pc_results[1].all_proposed_actions), 3)
        self.assertEqual(len(pc_results[2].all_proposed_actions), 3)

    def test_pc_number_of_actions_three_nodes(self):
        """
        test if the number of all proposed actions is correct
        """
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(1000)
        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        pc_results = tst.execute_pipeline_steps()
        self.assertEqual(len(pc_results[0].actions), 3)
        self.assertEqual(len(pc_results[1].actions), 0)
        self.assertEqual(len(pc_results[2].actions), 1)

    def test_pc_number_of_all_proposed_actions_four_nodes(self):
        """
        test if the number of all proposed actions is correct
        """
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
                SampleEdge(NodeReference("X"), NodeReference("W"), 7),
                SampleEdge(NodeReference("W"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(1000)
        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        pc_results = tst.execute_pipeline_steps()
        self.assertEqual(len(pc_results[0].all_proposed_actions), 6)
        self.assertEqual(len(pc_results[1].all_proposed_actions), 6)
        # TODO: think about whether the pairs with neighbours generator returns what we want, but the counting seems correct
        self.assertEqual(len(pc_results[3].all_proposed_actions), 7)

    def test_pc_calculate_pearson_correlations(self):
        """
        Test conditional independence of ordered pairs given pairs of other variables works.
        """
        algo = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[],
                name="PC",
            )
        )
        sample_generator = self._sample_generator()
        test_data, graph = sample_generator.generate(10000)
        tst = algo()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        reference = load_fixture_graph(
            "tests/fixtures/pc_e2e/pc_calculate_pearson_correlations.json"
        )
        # dump_fixture_graph(tst.graph, "fixtures/pc_e2e/pc_calculate_pearson_correlations.json")
        self.assertGraphStructureIsEqual(reference, tst.graph)

    def test_pc_correlation_coefficient_test(self):
        algo = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(),
                    CorrelationCoefficientTest(threshold=0.05),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[],
                name="PC",
            )
        )
        sample_generator = self._sample_generator()
        test_data, graph = sample_generator.generate(10000)
        tst = algo()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        reference = load_fixture_graph(
            "tests/fixtures/pc_e2e/pc_correlation_coefficient_test.json"
        )
        # dump_fixture_graph(tst.graph, "fixtures/pc_e2e/pc_correlation_coefficient_test.json")
        self.assertGraphStructureIsEqual(reference, tst.graph)

    def test_pc_partial_correlation_test(self):
        algo = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(),
                    CorrelationCoefficientTest(threshold=0.05),
                    PartialCorrelationTest(threshold=0.05),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[],
                name="PC",
            )
        )
        sample_generator = self._sample_generator()
        test_data, graph = sample_generator.generate(10000)
        tst = algo()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        reference = load_fixture_graph(
            "tests/fixtures/pc_e2e/pc_partial_correlation_test.json"
        )
        # dump_fixture_graph(tst.graph, "fixtures/pc_e2e/pc_partial_correlation_test.json")
        self.assertGraphStructureIsEqual(reference, tst.graph)

    def test_pc_extended_partial_correlation_test_matrix(self):
        algo = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(),
                    CorrelationCoefficientTest(threshold=0.05),
                    PartialCorrelationTest(threshold=0.05),
                    ExtendedPartialCorrelationTestMatrix(threshold=0.05),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[],
                name="PC",
            )
        )
        sample_generator = self._sample_generator()
        test_data, graph = sample_generator.generate(10000)
        tst = algo()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        reference = load_fixture_graph(
            "tests/fixtures/pc_e2e/pc_extended_partial_correlation_test_matrix.json"
        )
        # dump_fixture_graph(tst.graph, "fixtures/pc_e2e/pc_extended_partial_correlation_test_matrix.json")
        self.assertGraphStructureIsEqual(reference, tst.graph)

    def test_pc_collider_test(self):
        algo = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(),
                    CorrelationCoefficientTest(threshold=0.05),
                    PartialCorrelationTest(threshold=0.05),
                    ExtendedPartialCorrelationTestMatrix(threshold=0.05),
                    ColliderTest(display_name="Collider Test"),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[],
                name="PC",
            )
        )
        sample_generator = self._sample_generator()
        test_data, graph = sample_generator.generate(10000)
        tst = algo()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        reference = load_fixture_graph("tests/fixtures/pc_e2e/pc_collider_test.json")
        # dump_fixture_graph(tst.graph, "fixtures/pc_e2e/pc_collider_test.json")
        self.assertGraphStructureIsEqual(reference, tst.graph)

    def test_tracking_triples_two_nodes(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        pc_results = tst.execute_pipeline_steps()
        triples = []
        for result in pc_results:
            for action in result.all_proposed_actions:
                if "triple" in action.data:
                    triples.append(action.data["triple"])
        self.assertEqual(len(triples), 1)

    def test_tracking_triples_three_nodes(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        pc_results = tst.execute_pipeline_steps()

        triples = []
        for result in pc_results:
            for action in result.all_proposed_actions:
                if "triple" in action.data:
                    triples.append(action.data["triple"])
        self.assertEqual(len(triples), 6)

    def test_tracking_triples_four_nodes(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
                SampleEdge(NodeReference("X"), NodeReference("W"), 7),
                SampleEdge(NodeReference("W"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 6),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 5),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        pc_results = tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        triples = []
        for result in pc_results:
            for action in result.all_proposed_actions:
                if "triple" in action.data:
                    triples.append(action.data["triple"])
        # two out of four + two out of four times two given two possible conditioning nodes + two out of four times two beccause PC tests for neighbours of X and of Y.
        self.assertEqual(len(triples), 6 + 12 + 12)

    def test_track_triples_three_nodes_custom_pc(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        pc_results = tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        triples = []
        for result in pc_results:
            for proposed_action in result.all_proposed_actions:
                if "triple" in proposed_action.data:
                    triples.append(proposed_action.data["triple"])
        # two out of four + two out of four times two given two possible conditioning nodes + two out of four times two beccause PC tests for neighbours of X and of Y.
        self.assertIn(len(triples), [6, 7, 8])

    def test_track_triples_two_nodes_custom_pc_unconditionally_independent(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PCClassic()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        pc_results = tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        triples = []
        for result in pc_results:
            for proposed_action in result.all_proposed_actions:
                if "triple" in proposed_action.data:
                    triples.append(proposed_action.data["triple"])
        # two out of four + two out of four times two given two possible conditioning nodes + two out of four times two beccause PC tests for neighbours of X and of Y.
        self.assertEqual(len(triples), 3 + 2)

    def test_track_triples_three_nodes_pc_unconditionally_independent(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        pc_results = tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        triples = []
        for result in pc_results:
            for action in result.all_proposed_actions:
                if "triple" in action.data:
                    triples.append(action.data["triple"])
        # TODO: find issue with tracking in partial correlation test in this setting
        pass

    def test_d_separation_on_output_of_pc(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
            random=lambda: rdnv(0, 1),
        )
        test_data, graph = sample_generator.generate(10000)
        tst = PC()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)
        x = tst.graph.node_by_id("X")
        y = tst.graph.node_by_id("Y")
        z = tst.graph.node_by_id("Z")
        self.assertEqual(tst.graph.are_nodes_d_separated_cpdag(x, z, []), False)
        self.assertEqual(tst.graph.are_nodes_d_separated_cpdag(x, z, [y]), True)
