import itertools
import math
from typing import Tuple, List, Optional, Generic
import logging

import torch
from scipy import stats

from causy.generators import AllCombinationsGenerator, PairsWithNeighboursGenerator
from causy.math_utils import get_t_and_critical_t
from causy.interfaces import (
    PipelineStepInterface,
    BaseGraphInterface,
    NodeInterface,
    AS_MANY_AS_FIELDS,
    GeneratorInterface,
    PipelineStepInterfaceType,
)
from causy.models import ComparisonSettings, TestResultAction, TestResult
from causy.variables import IntegerParameter, BoolParameter

logger = logging.getLogger(__name__)


class CorrelationCoefficientTest(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    generator: Optional[GeneratorInterface] = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    chunk_size_parallel_processing: IntegerParameter = 1
    parallel: BoolParameter = False

    def process(
        self, nodes: List[str], graph: BaseGraphInterface
    ) -> Optional[TestResult]:
        """
        Test if u and v are independent and delete edge in graph if they are.
        :param nodes: list of nodes
        :return: A TestResult with the action to take
        """
        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        # Use Fisher's Z test for independency of u and v
        sample_size = len(x.values)
        corr = graph.edge_value(x, y)["correlation"]

        # Fisher's Z transformation
        fisher_z = 0.5 * math.log((1 + corr) / (1 - corr))

        # Use absolute value of fisher_z for a two-tailed test
        z_value = abs(fisher_z) * math.sqrt(sample_size - 3)

        # Compute the p-value
        p_value = 2 * (1 - stats.norm.cdf(z_value))

        # If the p value is smaller than the threshold, the null hypothesis (independence) is rejected, otherwise we accept it and delete the edge
        if p_value > self.threshold:
            logger.debug(f"Nodes {x.name} and {y.name} are uncorrelated")
            logger.debug(f"P-value: {p_value}")
            return TestResult(
                u=x,
                v=y,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={"separatedBy": [], "triple": [x, y, []]},
            )
        else:
            return TestResult(
                u=x,
                v=y,
                action=TestResultAction.DO_NOTHING,
                data={"triple": [x, y, []]},
            )


class PartialCorrelationTest(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    generator: Optional[GeneratorInterface] = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=3, max=3)
    )
    chunk_size_parallel_processing: IntegerParameter = 1
    parallel: BoolParameter = False

    def process(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> Optional[List[TestResult]]:
        """
        Test if nodes u,v are independent given node z based on a partial correlation test.
        We use this test for all combinations of 3 nodes because it is faster than the extended test (which supports combinations of n nodes). We can
        use it to remove edges between nodes which are not independent given another node and so reduce the number of combinations for the extended test.
        :param nodes: the nodes to test
        :return: A TestResult with the action to take

        TODO: we are testing (C and E given B) and (E and C given B), we just need one of these, remove redundant tests.
        """
        node_names = []
        for node in nodes:
            node_names.append(graph.nodes[node].name)
        results = []
        already_deleted_edges = set()
        for node in nodes:
            remaining_nodes = [n for n in nodes if n != node]
            x: NodeInterface = graph.nodes[remaining_nodes[0]]
            y: NodeInterface = graph.nodes[remaining_nodes[1]]
            z: NodeInterface = graph.nodes[node]

            if x is None or y is None or z is None:
                return

            if not graph.edge_exists(x, y) or (y, x) in already_deleted_edges:
                continue

            try:
                cor_xy = graph.edge_value(x, y)["correlation"]
                cor_xz = graph.edge_value(x, z)["correlation"]
                cor_yz = graph.edge_value(y, z)["correlation"]
            except (KeyError, TypeError):
                return

            numerator = cor_xy - cor_xz * cor_yz
            denominator = ((1 - cor_xz**2) * (1 - cor_yz**2)) ** 0.5

            # Avoid division by zero
            if denominator == 0:
                return

            par_corr = numerator / denominator

            # make t test for independency of u and v given z
            sample_size = len(x.values)
            nb_of_control_vars = len(nodes) - 2

            # Fisher's Z transformation
            fisher_z = 0.5 * math.log((1 + par_corr) / (1 - par_corr))

            # Use absolute value of fisher_z for a two-tailed test
            z_value = abs(fisher_z) * math.sqrt(sample_size - nb_of_control_vars - 3)

            # Compute the p-value
            p_value = 2 * (1 - stats.norm.cdf(z_value))

            # If the p value is smaller than the threshold, the null hypothesis (conditional independence) is rejected, otherwise we accept it and delete the edge
            if p_value > self.threshold:
                results.append(
                    TestResult(
                        u=x,
                        v=y,
                        action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                        data={"separatedBy": [z], "triple": [x, y, [z]]},
                    )
                )
                already_deleted_edges.add((x, y))

            else:
                results.append(
                    TestResult(
                        u=x,
                        v=y,
                        action=TestResultAction.DO_NOTHING,
                        data={"triple": [x, y, [z]]},
                    )
                )
        return results


class ExtendedPartialCorrelationTestMatrix(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    generator: Optional[GeneratorInterface] = PairsWithNeighboursGenerator(
        comparison_settings=ComparisonSettings(min=4, max=AS_MANY_AS_FIELDS),
        shuffle_combinations=False,
    )
    chunk_size_parallel_processing: IntegerParameter = 1000
    parallel: BoolParameter = False

    def process(
        self, nodes: List[str], graph: BaseGraphInterface
    ) -> Optional[TestResult]:
        """
        Test if nodes u,v are independent given Z (set of nodes) based on partial correlation using the inverted covariance matrix (precision matrix).
        https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
        We use this test for all combinations of more than 3 nodes because it is slower.
        If the covariance matrix is ill-conditioned, i.e., its condition number is high, the precision matrix is not reliable.
        In that case, we throw a warning.
        :param nodes: the nodes to test
        :return: A TestResult with the action to take
        """

        if not graph.edge_exists(graph.nodes[nodes[0]], graph.nodes[nodes[1]]):
            return

        other_neighbours = set(
            [
                k
                for k, value in graph.edges[nodes[0]].items()
                if graph.directed_edge_exists(k, nodes[0])
            ]
        )
        other_neighbours.remove(graph.nodes[nodes[1]].id)

        if not set(nodes[2:]).issubset(set([on for on in list(other_neighbours)])):
            return
        cov_matrix = torch.cov(
            torch.stack([graph.nodes[node].values for node in nodes])
        )
        # check if the covariance matrix is ill-conditioned
        if torch.det(cov_matrix) == 0:
            logger.warning(
                "The covariance matrix is ill-conditioned. The precision matrix is not reliable."
            )
            return

        inverse_cov_matrix = torch.inverse(cov_matrix)

        n = inverse_cov_matrix.size(0)
        diagonal = torch.diag(inverse_cov_matrix)
        diagonal_matrix = torch.zeros((n, n), dtype=torch.float64)
        for i in range(n):
            diagonal_matrix[i, i] = diagonal[i]

        helper = torch.mm(torch.sqrt(diagonal_matrix), inverse_cov_matrix)
        precision_matrix = torch.mm(helper, torch.sqrt(diagonal_matrix))

        par_corr = (-1 * precision_matrix[0][1]) / torch.sqrt(
            precision_matrix[0][0] * precision_matrix[1][1]
        )

        sample_size = len(graph.nodes[nodes[0]].values)
        nb_of_control_vars = len(nodes) - 2

        # Fisher's Z transformation
        fisher_z = 0.5 * math.log((1 + par_corr) / (1 - par_corr))

        # Use absolute value of fisher_z for a two-tailed test
        z_value = abs(fisher_z) * math.sqrt(sample_size - nb_of_control_vars - 3)

        # Compute the p-value
        p_value = 2 * (1 - stats.norm.cdf(z_value))

        # If the p value is smaller than the threshold, the null hypothesis (conditional independence) is rejected, otherwise we accept it and delete the edge
        nodes_set = set([graph.nodes[n] for n in nodes])
        if p_value > self.threshold:
            logger.debug(
                f"Nodes {graph.nodes[nodes[0]].name} and {graph.nodes[nodes[1]].name} are uncorrelated given nodes {','.join([graph.nodes[on].name for on in other_neighbours])}"
            )
            return TestResult(
                u=graph.nodes[nodes[0]],
                v=graph.nodes[nodes[1]],
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={
                    "separatedBy": list(
                        nodes_set - {graph.nodes[nodes[0]], graph.nodes[nodes[1]]}
                    ),
                    "triple": [
                        graph.nodes[nodes[0]],
                        graph.nodes[nodes[1]],
                        list(
                            nodes_set - {graph.nodes[nodes[0]], graph.nodes[nodes[1]]}
                        ),
                    ],
                },
            )
        else:
            return TestResult(
                u=graph.nodes[nodes[0]],
                v=graph.nodes[nodes[1]],
                action=TestResultAction.DO_NOTHING,
                data={
                    "triple": [
                        graph.nodes[nodes[0]],
                        graph.nodes[nodes[1]],
                        list(
                            nodes_set - {graph.nodes[nodes[0]], graph.nodes[nodes[1]]}
                        ),
                    ],
                },
            )


def partial_correlation_regression(x, y, z):
    """
    Compute the partial correlation coefficient between x and y controlling for other variables in z using linear regression.

    Arguments:
    x, y : torch.Tensor : Variables for which the partial correlation is computed.
    z : torch.Tensor : Other variables used to control for in the partial correlation.

    Returns:
    partial_corr : torch.Tensor : Partial correlation coefficient between x and y.
    """
    # Define linear regression model
    model_x = torch.linalg.lstsq(z.T, x).solution
    model_y = torch.linalg.lstsq(z.T, y).solution

    residual_x = x - torch.matmul(model_x, z)
    residual_y = y - torch.matmul(model_y, z)

    # Compute correlation of residuals
    return torch.dot(residual_x, residual_y) / (
        torch.norm(residual_x) * torch.norm(residual_y)
    )


class ExtendedPartialCorrelationTestLinearRegression(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    generator: Optional[GeneratorInterface] = PairsWithNeighboursGenerator(
        comparison_settings=ComparisonSettings(min=4, max=AS_MANY_AS_FIELDS),
        shuffle_combinations=False,
    )
    chunk_size_parallel_processing: IntegerParameter = 1000
    parallel: BoolParameter = False

    def process(
        self, nodes: List[str], graph: BaseGraphInterface
    ) -> Optional[TestResult]:
        if not graph.edge_exists(graph.nodes[nodes[0]], graph.nodes[nodes[1]]):
            return

        other_neighbours = set(graph.edges[nodes[0]])
        other_neighbours.remove(graph.nodes[nodes[1]].id)

        partial_correlation = partial_correlation_regression(
            graph.nodes[nodes[0]].values,
            graph.nodes[nodes[1]].values,
            torch.stack([graph.nodes[node].values for node in nodes[2:]]),
        )

        sample_size = len(graph.nodes[nodes[0]].values)
        nb_of_control_vars = len(nodes) - 2

        t, critical_t = get_t_and_critical_t(
            sample_size,
            nb_of_control_vars,
            partial_correlation.item(),
            self.threshold,
        )
        nodes_set = set([graph.nodes[n] for n in nodes])
        if abs(t) < critical_t:
            logger.debug(
                f"Nodes {graph.nodes[nodes[0]].name} and {graph.nodes[nodes[1]].name} are uncorrelated given nodes {','.join([graph.nodes[on].name for on in other_neighbours])}"
            )
            return TestResult(
                u=graph.nodes[nodes[0]],
                v=graph.nodes[nodes[1]],
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={
                    "separatedBy": list(
                        nodes_set - {graph.nodes[nodes[0]], graph.nodes[nodes[1]]}
                    ),
                    "triple": [
                        graph.nodes[nodes[0]],
                        graph.nodes[nodes[1]],
                        list(
                            nodes_set - {graph.nodes[nodes[0]], graph.nodes[nodes[1]]}
                        ),
                    ],
                },
            )
        else:
            return TestResult(
                u=graph.nodes[nodes[0]],
                v=graph.nodes[nodes[1]],
                action=TestResultAction.DO_NOTHING,
                data={
                    "triple": [
                        graph.nodes[nodes[0]],
                        graph.nodes[nodes[1]],
                        list(
                            nodes_set - {graph.nodes[nodes[0]], graph.nodes[nodes[1]]}
                        ),
                    ]
                },
            )
