from typing import Tuple

import torch

from causy.generators import AllCombinationsGenerator
from causy.interfaces import (
    PipelineStepInterface,
    ComparisonSettings,
    BaseGraphInterface,
    TestResult,
    TestResultAction,
)


class CalculatePearsonCorrelations(PipelineStepInterface):
    generator = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    chunk_size_parallel_processing = 1
    parallel = False

    def test(self, nodes: Tuple[str], graph: BaseGraphInterface, result_queue):
        """
        Calculate the correlation between each pair of nodes and store it to the respective edge.
        :param nodes: list of nodes
        :param result_queue: the result queue to put the result in
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

        result_queue.put(
            TestResult(
                x=x,
                y=y,
                action=TestResultAction.UPDATE_EDGE,
                data=edge_value,
            )
        )