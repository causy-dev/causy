from causy.causal_discovery.constraint.algorithms.pc import PC_EDGE_TYPES, PC
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.graph_model import graph_model_factory
from causy.causal_discovery.constraint.independence_tests.common import (
    CorrelationCoefficientTest,
    PartialCorrelationTest,
    ExtendedPartialCorrelationTestMatrix,
)
from causy.models import Algorithm
from causy.causal_discovery.constraint.orientation_rules.pc import ColliderTest
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference

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
        # this should be 3, but with the current combinations generator, it does 4 tests
        self.assertEqual(len(pc_results[2].all_proposed_actions), 4)

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
