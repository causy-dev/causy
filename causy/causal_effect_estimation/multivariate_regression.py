from typing import Tuple

import torch

from causy.generators import PairsWithEdgesInBetweenGenerator
from causy.interfaces import (
    BaseGraphInterface,
    TestResult,
    ComparisonSettings,
    PipelineStepInterface,
    TestResultAction,
)


class ComputeDirectEffectsMultivariateRegression(PipelineStepInterface):
    generator = PairsWithEdgesInBetweenGenerator()

    def test(self, nodes: Tuple[str], graph: BaseGraphInterface) -> TestResult:
        """
        Calculate the direct effect of each edge in the graph using multivariate regression.
        :param nodes: list of nodes
        :return: A TestResult with the action to take
        """
        effect_variable = graph.nodes[nodes[1]]
        cause_variable = graph.nodes[nodes[0]]

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
        print(effect_variable_parents_values.shape)
        # multivariate regression via tridiagonal reduction and SVD, see torch documentation
        # TODO: We compute all edge weights that have ingoing edges to variables with several parents multiple times (once for each parent)
        #       We should cache the results of the regression for each variable with multiple parents
        reshaped_effect_variable_values = effect_variable.values.view(-1, 1)
        print(reshaped_effect_variable_values.shape)
        coefficients = torch.linalg.lstsq(
            reshaped_effect_variable_values,
            effect_variable_parents_values,
            driver="gelsd",
        ).solution
        print(coefficients)
        edge_data["direct_effect"] = coefficients[0][0].item()

        return TestResult(
            u=graph.nodes[nodes[0]],
            v=graph.nodes[nodes[1]],
            action=TestResultAction.UPDATE_EDGE_DIRECTED,
            data=edge_data,
        )
