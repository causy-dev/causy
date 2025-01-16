from causy.causal_discovery.constraint.algorithms.pc import PC_EDGE_TYPES
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.generators import PairsWithNeighboursGenerator, AllCombinationsGenerator
from causy.graph_model import graph_model_factory
from causy.causal_discovery.constraint.independence_tests.common import (
    CorrelationCoefficientTest,
    PartialCorrelationTest,
)
from causy.models import ComparisonSettings, Algorithm
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from tests.utils import CausyTestCase


class GeneratorsTestCase(CausyTestCase):
    SEED = 1

    # TODO, wip
    def test_pairs_with_neighbours_generator_two_nodes(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 1),
            ],
            random=lambda: rdnv(0, 1),
        )

        algo = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(),
                    CorrelationCoefficientTest(threshold=0.005),
                    PartialCorrelationTest(threshold=0.005),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[],
                name="PC",
            )
        )
        test_data, graph = sample_generator.generate(1000)
        tst = algo()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        result = PairsWithNeighboursGenerator(
            comparison_settings=ComparisonSettings(min=2, max=4)
        ).generate(tst.graph.graph, tst)
        all_results = []

        for i in result:
            all_results.append(i)
        pass

    def test_pairs_with_neighbours_generator_three_nodes_one_neighbour(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 1),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 1),
            ],
            random=lambda: rdnv(0, 1),
        )

        algo = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(),
                    CorrelationCoefficientTest(threshold=0.005),
                    PartialCorrelationTest(threshold=0.005),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[],
                name="PC",
            )
        )
        test_data, graph = sample_generator.generate(1000)
        tst = algo()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        result = PairsWithNeighboursGenerator(
            comparison_settings=ComparisonSettings(min=3, max=4)
        ).generate(tst.graph.graph, tst)
        all_results = []

        for i in result:
            all_results.append(i)
        pass

    def test_pairs_with_neighbours_generator_three_nodes_two_neighbours(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 1),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 1),
            ],
            random=lambda: rdnv(0, 1),
        )

        algo = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(),
                    CorrelationCoefficientTest(threshold=0.005),
                    PartialCorrelationTest(threshold=0.005),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[],
                name="PC",
            )
        )
        test_data, graph = sample_generator.generate(1000)
        tst = algo()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        result = PairsWithNeighboursGenerator(
            comparison_settings=ComparisonSettings(min=3, max=4)
        ).generate(tst.graph.graph, tst)
        all_results = []

        for i in result:
            all_results.append(i)
        pass

    def test_pairs_with_neighbours_generator_four_nodes_fully_connected(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 1),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("X"), NodeReference("W"), 1),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("Y"), NodeReference("W"), 1),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 1),
            ],
            random=lambda: rdnv(0, 1),
        )

        algo = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(),
                    CorrelationCoefficientTest(threshold=0.005),
                    PartialCorrelationTest(threshold=0.005),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[],
                name="PC",
            )
        )
        test_data, graph = sample_generator.generate(1000)
        tst = algo()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        result = PairsWithNeighboursGenerator(
            comparison_settings=ComparisonSettings(min=3, max=4)
        ).generate(tst.graph.graph, tst)
        all_results = []

        for i in result:
            all_results.append(i)
        pass

    def test_all_combinations_generator_two_nodes(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 1),
            ],
            random=lambda: rdnv(0, 1),
        )

        algo = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(),
                    CorrelationCoefficientTest(threshold=0.005),
                    PartialCorrelationTest(threshold=0.005),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[],
                name="PC",
            )
        )

        test_data, graph = sample_generator.generate(1000)
        tst = algo()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        result = AllCombinationsGenerator(
            comparison_settings=ComparisonSettings(min=2, max=2)
        ).generate(tst.graph.graph, tst)
        all_results = []

        for i in result:
            all_results.append(i)
        pass

    def test_all_combinations_generator(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 1),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 1),
            ],
            random=lambda: rdnv(0, 1),
        )

        algo = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(),
                    CorrelationCoefficientTest(threshold=0.005),
                    PartialCorrelationTest(threshold=0.005),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[],
                name="PC",
            )
        )
        test_data, graph = sample_generator.generate(1000)
        tst = algo()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        result = AllCombinationsGenerator(
            comparison_settings=ComparisonSettings(min=3, max=3)
        ).generate(tst.graph.graph, tst)
        all_results = []

        for i in result:
            all_results.append(i)

        pass

    def test_all_combinations_generator_four_nodes_fully_connected(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 1),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("X"), NodeReference("W"), 1),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("Y"), NodeReference("W"), 1),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 1),
            ],
            random=lambda: rdnv(0, 1),
        )

        algo = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(),
                    CorrelationCoefficientTest(threshold=0.005),
                    PartialCorrelationTest(threshold=0.005),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[],
                name="PC",
            )
        )
        test_data, graph = sample_generator.generate(1000)
        tst = algo()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        result = AllCombinationsGenerator(
            comparison_settings=ComparisonSettings(min=2, max=4)
        ).generate(tst.graph.graph, tst)
        all_results = []

        for i in result:
            all_results.append(i)
        pass
