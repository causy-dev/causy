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
)
from causy.algorithms.fci import FCIEdgeType
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference

from tests.utils import CausyTestCase


class IndependenceTestTestCase(CausyTestCase):
    SEED = 42

    def test_correlation_coefficient_test(self):
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
            ],
            random=lambda: torch.normal(0, 1, (1, 1)),
        )

        data, _ = model.generate(1000)

        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.1),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # X and Y are dependent
        self.assertEqual(len(tst.graph.action_history[-1]["actions"]), 0)

    def test_correlation_coefficient_test_collider(self):
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Z"), 3),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 3),
            ],
            random=lambda: torch.normal(0, 1, (1, 1)),
        )

        data, _ = model.generate(1000)

        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.1),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # X and Y are independent
        self.assertEqual(len(tst.graph.action_history[-1]["actions"]), 1)

    def test_partial_correlation_test(self):
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
            ]
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
        print(tst.graph.action_history[-1]["actions"])
        # X and Z are independent given Y, no other pair of nodes is independent given one other node
        self.assertEqual(len(tst.graph.action_history[-1]["actions"]), 1)

    def test_extended_partial_correlation_test_matrix(self):
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
                SampleEdge(NodeReference("X"), NodeReference("W"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 8),
            ]
        )
        data, graph = model.generate(10000)
        pipeline = [
            CalculatePearsonCorrelations(),
            ExtendedPartialCorrelationTestMatrix(threshold=0.01),
        ]
        tst = graph_model_factory(pipeline_steps=pipeline)()
        tst.create_graph_from_data(data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        # X and Z are independent given Y and W, no other pair of nodes is independent given two other nodes
        self.assertEqual(len(tst.graph.action_history[-1]["actions"]), 1)
        print(retrieve_edges(tst.graph))
        self.assertGraphStructureIsEqual(tst.graph, graph)

    def test_combinations_1(self):
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
                SampleEdge(NodeReference("X"), NodeReference("W"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 8),
            ],
            random=lambda: random.normalvariate(0, 1),
        )
        data, graph = model.generate(100000)
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
