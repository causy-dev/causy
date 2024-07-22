from typing import List, Optional

from causy.causal_discovery.constraint.algorithms.pc import PC_EDGE_TYPES
from causy.causal_discovery.constraint.independence_tests.common import (
    CorrelationCoefficientTest,
)
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.interfaces import GraphUpdateHook, BaseGraphInterface, TestResultInterface
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from causy.models import Algorithm
from causy.graph_model import graph_model_factory

from tests.utils import CausyTestCase


class RemoveAllActionsHook(GraphUpdateHook):
    def execute(
        self, graph: BaseGraphInterface, updates: List[TestResultInterface]
    ) -> Optional[List[TestResultInterface]]:
        return []


class HookTestException(Exception):
    pass


class ThrowExceptionHook(GraphUpdateHook):
    def execute(
        self, graph: BaseGraphInterface, updates: List[TestResultInterface]
    ) -> Optional[List[TestResultInterface]]:
        raise HookTestException("This is a test exception")


class HooksTestCase(CausyTestCase):
    def test_pre_graph_update_hook(self):
        samples = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
            ]
        )

        data, graph = samples.generate(100)
        algorithm = Algorithm(
            pipeline_steps=[
                CalculatePearsonCorrelations(),
                CorrelationCoefficientTest(threshold=0.05),
            ],
            name="pc",
            edge_types=PC_EDGE_TYPES,
            extensions=[],
            pre_graph_update_hooks=[RemoveAllActionsHook()],
        )

        model = graph_model_factory(algorithm)()
        model.create_graph_from_data(data)
        model.create_all_possible_edges()
        model.execute_pipeline_steps()
        self.assertEqual(len(model.graph.action_history[0].actions), 0)

    def test_post_graph_update_hook(self):
        samples = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
            ]
        )

        data, graph = samples.generate(100)
        algorithm = Algorithm(
            pipeline_steps=[
                CalculatePearsonCorrelations(),
                CorrelationCoefficientTest(threshold=0.05),
            ],
            name="pc",
            edge_types=PC_EDGE_TYPES,
            extensions=[],
            post_graph_update_hooks=[ThrowExceptionHook()],
        )

        model = graph_model_factory(algorithm)()
        model.create_graph_from_data(data)
        model.create_all_possible_edges()
        with self.assertRaises(HookTestException):
            model.execute_pipeline_steps()
