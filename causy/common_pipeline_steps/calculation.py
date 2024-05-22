from typing import Tuple, Optional, Generic

import torch

from causy.generators import AllCombinationsGenerator
from causy.interfaces import (
    PipelineStepInterface,
    BaseGraphInterface,
    GeneratorInterface,
    PipelineStepInterfaceType,
)
from causy.models import ComparisonSettings, TestResultAction, TestResult


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

        x_val = x.values
        y_val = y.values

        cov_xy = torch.mean((x_val - x_val.mean()) * (y_val - y_val.mean()))
        std_x = x_val.std(unbiased=False)
        std_y = y_val.std(unbiased=False)
        pearson_correlation = cov_xy / (std_x * std_y)

        edge_value["correlation"] = pearson_correlation.item()

        return TestResult(
            u=x,
            v=y,
            action=TestResultAction.UPDATE_EDGE,
            data=edge_value,
        )
