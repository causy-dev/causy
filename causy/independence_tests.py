import itertools
from typing import Tuple, List, Optional
import logging

import torch

from causy.generators import AllCombinationsGenerator, PairsWithNeighboursGenerator
from causy.utils import get_t_and_critical_t, pearson_correlation
from causy.interfaces import (
    IndependenceTestInterface,
    BaseGraphInterface,
    NodeInterface,
    TestResult,
    TestResultAction,
    AS_MANY_AS_FIELDS,
    ComparisonSettings,
)

logger = logging.getLogger(__name__)


class CalculateCorrelations(IndependenceTestInterface):
    GENERATOR = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    CHUNK_SIZE_PARALLEL_PROCESSING = 1
    PARALLEL = False

    def test(self, nodes: Tuple[str], graph: BaseGraphInterface) -> TestResult:
        """
        Calculate the correlation between each pair of nodes and store it to the respective edge.
        :param nodes: list of nodes
        :return: A TestResult with the action to take
        """
        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]
        edge_value = graph.edge_value(graph.nodes[nodes[0]], graph.nodes[nodes[1]])
        edge_value["correlation"] = pearson_correlation(
            x.values,
            y.values,
        ).item()
        return TestResult(
            x=x,
            y=y,
            action=TestResultAction.UPDATE_EDGE,
            data=edge_value,
        )


class CorrelationCoefficientTest(IndependenceTestInterface):
    GENERATOR = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    CHUNK_SIZE_PARALLEL_PROCESSING = 1
    PARALLEL = False

    def test(self, nodes: List[str], graph: BaseGraphInterface) -> Optional[TestResult]:
        """
        Test if x and y are independent and delete edge in graph if they are.
        :param nodes: list of nodes
        :return: A TestResult with the action to take
        """
        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        # make t test for independency of x and y
        sample_size = len(x.values)
        nb_of_control_vars = 0
        corr = graph.edge_value(x, y)["correlation"]
        t, critical_t = get_t_and_critical_t(
            sample_size, nb_of_control_vars, corr, self.threshold
        )
        if abs(t) < critical_t:
            logger.debug(f"Nodes {x.name} and {y.name} are uncorrelated")
            return TestResult(
                x=x,
                y=y,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={},
            )


class PartialCorrelationTest(IndependenceTestInterface):
    GENERATOR = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=3, max=3)
    )
    CHUNK_SIZE_PARALLEL_PROCESSING = 1
    PARALLEL = False

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> Optional[List[TestResult]]:
        """
        Test if nodes x,y are independent given node z based on a partial correlation test.
        We use this test for all combinations of 3 nodes because it is faster than the extended test (which supports combinations of n nodes). We can
        use it to remove edges between nodes which are not independent given another node and so reduce the number of combinations for the extended test.
        :param nodes: the nodes to test
        :return: A TestResult with the action to take

        TODO: we are testing (C and E given B) and (E and C given B), we just need one of these, remove redundant tests.
        """
        results = []
        already_deleted_edges = set()
        for nodes in itertools.permutations(nodes):
            x: NodeInterface = graph.nodes[nodes[0]]
            y: NodeInterface = graph.nodes[nodes[1]]
            z: NodeInterface = graph.nodes[nodes[2]]

            # Avoid division by zero
            if x is None or y is None or z is None:
                return

            if not graph.edge_exists(x, y) or (y, x) in already_deleted_edges:
                continue

            try:
                cor_xy = graph.edge_value(x, y)["correlation"]
                cor_xz = graph.edge_value(x, z)["correlation"]
                cor_yz = graph.edge_value(y, z)["correlation"]
            except KeyError:
                return

            numerator = cor_xy - cor_xz * cor_yz
            denominator = ((1 - cor_xz**2) * (1 - cor_yz**2)) ** 0.5

            # Avoid division by zero
            if denominator == 0:
                return

            par_corr = numerator / denominator

            # make t test for independency of x and y given z
            sample_size = len(x.values)
            nb_of_control_vars = len(nodes) - 2
            t, critical_t = get_t_and_critical_t(
                sample_size, nb_of_control_vars, par_corr, self.threshold
            )

            if abs(t) < critical_t:
                logger.debug(
                    f"Nodes {x.name} and {y.name} are uncorrelated given {z.name}"
                )

                results.append(
                    TestResult(
                        x=x,
                        y=y,
                        action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                        data={"separatedBy": [z]},
                    )
                )
                already_deleted_edges.add((x, y))
        return results


class ExtendedPartialCorrelationTestMatrix(IndependenceTestInterface):
    GENERATOR = PairsWithNeighboursGenerator(
        comparison_settings=ComparisonSettings(min=4, max=AS_MANY_AS_FIELDS)
    )
    CHUNK_SIZE_PARALLEL_PROCESSING = 1000
    PARALLEL = False

    def test(self, nodes: List[str], graph: BaseGraphInterface) -> Optional[TestResult]:
        """
        Test if nodes x,y are independent given Z (set of nodes) based on partial correlation using the inverted covariance matrix (precision matrix).
        https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
        We use this test for all combinations of more than 3 nodes because it is slower.
        :param nodes: the nodes to test
        :return: A TestResult with the action to take
        """

        if not graph.edge_exists(graph.nodes[nodes[0]], graph.nodes[nodes[1]]):
            return

        other_neighbours = set(graph.edges[nodes[0]])
        other_neighbours.remove(graph.nodes[nodes[1]].id)

        if not set(nodes[2:]).issubset(set([on for on in list(other_neighbours)])):
            return

        inverse_cov_matrix = torch.inverse(
            torch.cov(torch.stack([graph.nodes[node].values for node in nodes]))
        )
        n = inverse_cov_matrix.size(0)
        diagonal = torch.diag(inverse_cov_matrix)
        diagonal_matrix = torch.zeros((n, n), dtype=torch.float64)
        for i in range(n):
            diagonal_matrix[i, i] = diagonal[i]

        helper = torch.mm(torch.sqrt(diagonal_matrix), inverse_cov_matrix)
        precision_matrix = torch.mm(helper, torch.sqrt(diagonal_matrix))

        sample_size = len(graph.nodes[nodes[0]].values)
        nb_of_control_vars = len(nodes) - 2

        t, critical_t = get_t_and_critical_t(
            sample_size,
            nb_of_control_vars,
            (
                (-1 * precision_matrix[0][1])
                / torch.sqrt(precision_matrix[0][0] * precision_matrix[1][1])
            ).item(),
            self.threshold,
        )

        if abs(t) < critical_t:
            logger.debug(
                f"Nodes {graph.nodes[nodes[0]].name} and {graph.nodes[nodes[1]].name} are uncorrelated given nodes {','.join([graph.nodes[on].name for on in other_neighbours])}"
            )
            nodes_set = set([graph.nodes[n] for n in nodes])
            return TestResult(
                x=graph.nodes[nodes[0]],
                y=graph.nodes[nodes[1]],
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={
                    "separatedBy": list(
                        nodes_set - {graph.nodes[nodes[0]], graph.nodes[nodes[1]]}
                    )
                },
            )


class PlaceholderTest(IndependenceTestInterface):
    NUM_OF_COMPARISON_ELEMENTS = 2
    CHUNK_SIZE_PARALLEL_PROCESSING = 10
    PARALLEL = False

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> List[TestResult] | TestResult:
        """
        Placeholder test for testing purposes
        :param nodes:
        :param graph:
        :return:
        """
        logger.debug(f"PlaceholderTest {nodes}")
        return TestResult(x=None, y=None, action=TestResultAction.DO_NOTHING, data={})
