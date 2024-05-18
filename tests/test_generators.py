from causy.algorithms.pc import PC_EDGE_TYPES
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.generators import PairsWithNeighboursGenerator
from causy.graph_model import graph_model_factory
from causy.graph_utils import retrieve_edges
from causy.independence_tests.common import (
    CorrelationCoefficientTest,
    PartialCorrelationTest,
)
from causy.interfaces import CausyAlgorithm, ComparisonSettings
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from tests.utils import CausyTestCase


class GeneratorsTestCase(CausyTestCase):
    SEED = 1

    def test_pairs_with_neighbours_generator(self):
        rdnv = self.seeded_random.normalvariate
        sample_generator = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 1),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 1),
                SampleEdge(NodeReference("Y"), NodeReference("W"), 1),
            ],
            random=lambda: rdnv(0, 1),
        )

        algo = graph_model_factory(
            CausyAlgorithm(
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
            all_results.extend(i)
