import random

import torch

from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.graph_model import graph_model_factory
from causy.graph_utils import retrieve_edges
from causy.math_utils import sum_lists
from causy.independence_tests.common import (
    CorrelationCoefficientTest,
    PartialCorrelationTest,
    ExtendedPartialCorrelationTestMatrix,
    ExtendedPartialCorrelationTestLinearRegression,
)
from causy.algorithms.fci import FCIEdgeType
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference

from tests.utils import CausyTestCase


class IndependenceTestTestCase(CausyTestCase):
    SEED = 99

    def test_correlation_coefficient_test(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
            ],
            random=lambda: rdnv(0, 1),
        )

        data, graph = model.generate(1000)

        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.1),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # X and Y are dependent
        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_correlation_coefficient_test_2(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("Y"), NodeReference("X"), 1),
            ],
            random=lambda: rdnv(0, 1),
        )

        data, graph = model.generate(1000000)

        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.1),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # X and Y are dependent
        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_correlation_coefficient_test_collider(self):
        rdnv = self.seeded_random.normalvariate

        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Z"), 3),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 3),
            ],
            random=lambda: rdnv(0, 1),
        )

        data, graph = model.generate(1000)

        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.1),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # X and Y are independent
        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_partial_correlation_test(self):
        rdnv = self.seeded_random.normalvariate

        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
            ],
            random=lambda: rdnv(0, 1),
        )
        data, _ = model.generate(1000)

        pipeline = [
            CalculatePearsonCorrelations(),
            PartialCorrelationTest(threshold=0.01),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # X and Z are independent given Y, no other pair of nodes is independent given one other node
        self.assertEqual(len(tst.graph.action_history[-1]["actions"]), 1)

    def test_partial_correlation_test_2(self):
        rdnv = self.seeded_random.normalvariate

        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("L"), 6),
                SampleEdge(NodeReference("L"), NodeReference("Z"), 7),
            ],
            random=lambda: rdnv(0, 1),
        )
        data, graph = model.generate(1000)

        pipeline = [
            CalculatePearsonCorrelations(),
            PartialCorrelationTest(threshold=0.01),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # X and Z are independent given Y, no other pair of nodes is independent given one other node
        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_partial_correlation_test_collider(self):
        rdnv = self.seeded_random.normalvariate

        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Z"), 3),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 3),
                SampleEdge(NodeReference("X"), NodeReference("Y"), 3),
            ],
            random=lambda: rdnv(0, 1),
        )

        data, graph = model.generate(1000)

        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.1),
            PartialCorrelationTest(threshold=0.01),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # X and Y are independent
        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_extended_partial_correlation_test_matrix(self):
        rdnv = self.seeded_random.normalvariate

        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
                SampleEdge(NodeReference("X"), NodeReference("W"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 8),
            ],
            random=lambda: rdnv(0, 1),
        )
        data, graph = model.generate(1000000)

        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.1),
            PartialCorrelationTest(threshold=0.01),
            ExtendedPartialCorrelationTestMatrix(threshold=0.01),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # X and Z are independent given Y and W, no other pair of nodes is independent given two other nodes
        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_extended_partial_correlation_test_matrix2(self):
        rdnv = self.seeded_random.normalvariate
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
                SampleEdge(NodeReference("X"), NodeReference("W"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 8),
                SampleEdge(NodeReference("X"), NodeReference("Q"), 3),
                SampleEdge(NodeReference("Q"), NodeReference("Z"), 4),
            ],
            random=lambda: rdnv(0, 1),
        )
        data, graph = model.generate(1000000)

        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.01),
            PartialCorrelationTest(threshold=0.01),
            ExtendedPartialCorrelationTestMatrix(threshold=0.01),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # Y and W are independent given X, X and Z are independent given Y and W
        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_extended_partial_correlation_test_matrix3(self):
        rdnv = self.seeded_random.normalvariate

        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
                SampleEdge(NodeReference("X"), NodeReference("W"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 8),
                SampleEdge(NodeReference("X"), NodeReference("Q"), 3),
                SampleEdge(NodeReference("Q"), NodeReference("Z"), 4),
                SampleEdge(NodeReference("X"), NodeReference("L"), 3),
                SampleEdge(NodeReference("L"), NodeReference("Z"), 4),
            ],
            random=lambda: rdnv(0, 1),
        )
        data, graph = model.generate(1000000)

        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.01),
            PartialCorrelationTest(threshold=0.01),
            ExtendedPartialCorrelationTestMatrix(threshold=0.01),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # Y and W are independent given X, X and Z are independent given Y and W
        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_extended_partial_correlation_test_linear_regression2(self):
        rdnv = self.seeded_random.normalvariate

        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
                SampleEdge(NodeReference("X"), NodeReference("W"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 8),
                SampleEdge(NodeReference("X"), NodeReference("Q"), 3),
                SampleEdge(NodeReference("Q"), NodeReference("Z"), 4),
            ],
            random=lambda: rdnv(0, 1),
        )
        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.01),
            PartialCorrelationTest(threshold=0.01),
            ExtendedPartialCorrelationTestLinearRegression(threshold=0.01),
        ]
        data, graph = model.generate(1000000)

        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # Y and W are independent given X, X and Z are independent given Y and W
        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_extended_partial_correlation_test_linear_regression3(self):
        rdnv = self.seeded_random.normalvariate

        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
                SampleEdge(NodeReference("X"), NodeReference("W"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 8),
                SampleEdge(NodeReference("X"), NodeReference("Q"), 3),
                SampleEdge(NodeReference("Q"), NodeReference("Z"), 4),
                SampleEdge(NodeReference("X"), NodeReference("L"), 3),
                SampleEdge(NodeReference("L"), NodeReference("Z"), 4),
            ],
            random=lambda: rdnv(0, 1),
        )
        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.1),
            PartialCorrelationTest(threshold=0.1),
            ExtendedPartialCorrelationTestLinearRegression(threshold=0.1),
        ]
        data, graph = model.generate(1000000)

        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # Y and W are independent given X, X and Z are independent given Y and W
        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_extended_partial_correlation_test_linear_regression(self):
        rdnv = self.seeded_random.normalvariate

        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
                SampleEdge(NodeReference("X"), NodeReference("W"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 8),
            ],
            random=lambda: rdnv(0, 1),
        )
        data, graph = model.generate(1000000)

        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.1),
            PartialCorrelationTest(threshold=0.01),
            ExtendedPartialCorrelationTestLinearRegression(threshold=0.01),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # X and Z are independent given Y and W, no other pair of nodes is independent given two other nodes
        self.assertGraphStructureIsEqual(tst.graph, graph)
