from typing import Tuple, Optional, Generic

import torch

from causy.causal_discovery.constraint.independence_tests.conditional_independence_calculations import (
    PearsonStudentsTTest,
)
from causy.generators import AllCombinationsGenerator
from causy.interfaces import (
    PipelineStepInterface,
    BaseGraphInterface,
    GeneratorInterface,
    PipelineStepInterfaceType,
)
from causy.models import ComparisonSettings, TestResultAction, TestResult
from causy.variables import CausyObjectParameter


class CalculateEdgeCorrelations(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    generator: Optional[GeneratorInterface] = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    chunk_size_parallel_processing: int = 1
    parallel: bool = False
    conditional_independence_test: CausyObjectParameter = PearsonStudentsTTest()

    def process(self, nodes: Tuple[str], graph: BaseGraphInterface) -> TestResult:
        """
        Test if u and v are independent and delete edge in graph if they are.
        :param nodes: list of nodes
        :return: A TestResult with the action to take
        """
        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]
        edge_value = graph.edge_value(graph.nodes[nodes[0]], graph.nodes[nodes[1]])
        correlation = self.conditional_independence_test.calculate_correlation(x, y, [])
        edge_value["correlation"] = correlation.item()
        return TestResult(
            u=x,
            v=y,
            action=TestResultAction.UPDATE_EDGE,
            data=edge_value,
        )


class CalculatePearsonCorrelations(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    generator: Optional[GeneratorInterface] = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    chunk_size_parallel_processing: int = 1
    parallel: bool = False

    def process(self, nodes: Tuple[str], graph: BaseGraphInterface) -> TestResult:
        """
        Calculate the correlation between each pair of nodes and store it to the respective edge.
        :param nodes: list of nodes
        :return: A TestResult with the action to take
        """
        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]
        edge_value = graph.edge_value(graph.nodes[nodes[0]], graph.nodes[nodes[1]])

        correlation = PearsonStudentsTTest.calculate_correlation(x, y, [])

        edge_value["correlation"] = correlation.item()

        return TestResult(
            u=x,
            v=y,
            action=TestResultAction.UPDATE_EDGE,
            data=edge_value,
        )
