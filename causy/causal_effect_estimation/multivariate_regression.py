from typing import Tuple, Optional

import torch

from causy.generators import PairsWithEdgesInBetweenGenerator
from causy.interfaces import (
    BaseGraphInterface,
    PipelineStepInterface,
    GeneratorInterface,
)
from causy.models import TestResultAction, TestResult


class ComputeDirectEffectsMultivariateRegression(PipelineStepInterface):
    generator: Optional[GeneratorInterface] = PairsWithEdgesInBetweenGenerator()

    chunk_size_parallel_processing: int = 1
    parallel: bool = False

    def process(self, nodes: Tuple[str], graph: BaseGraphInterface) -> TestResult:
        """
        Calculate the direct effect of each edge in the graph using multivariate regression.
        :param nodes: list of nodes
        :return: A TestResult with the action to take
        """
        effect_variable = graph.nodes[nodes[1]]
        cause_variable = graph.nodes[nodes[0]]

        if graph.undirected_edge_exists(cause_variable, effect_variable):
            # if the edge is not undirected, we do not need to calculate the direct effect
            return

        edge_data = graph.edge_value(cause_variable, effect_variable)

        all_parents_of_effect_variable = graph.parents_of_node(effect_variable)
        parents_of_effect_variable_without_cause_variable = [
            parent
            for parent in all_parents_of_effect_variable
            if parent != cause_variable
        ]

        effect_variable_parents_values = torch.stack(
            [cause_variable.values]
            + [
                parents_of_effect_variable_without_cause_variable[i].values
                for i in range(len(parents_of_effect_variable_without_cause_variable))
            ],
            dim=1,
        )
        # multivariate regression via tridiagonal reduction and SVD, see torch documentation
        reshaped_effect_variable_values = effect_variable.values.view(-1, 1)

        coefficients = torch.linalg.lstsq(
            effect_variable_parents_values,
            reshaped_effect_variable_values,
            driver="gelsd",
        ).solution

        edge_data["direct_effect"] = coefficients[0][0].item()

        return TestResult(
            u=graph.nodes[nodes[0]],
            v=graph.nodes[nodes[1]],
            action=TestResultAction.UPDATE_EDGE_DIRECTED,
            data=edge_data,
        )
