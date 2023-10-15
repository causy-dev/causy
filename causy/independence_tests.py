import itertools
from statistics import correlation, covariance  # , linear_regression
from typing import Tuple, List
import math

from causy.generators import AllCombinationsGenerator, PairsWithNeighboursGenerator

# Use cupy for GPU support - if available - otherwise use numpy
try:
    import cupy as np
except ImportError:
    import numpy as np

from causy.utils import get_t_and_critial_t, get_correlation

import logging

logger = logging.getLogger(__name__)

from causy.interfaces import (
    IndependenceTestInterface,
    BaseGraphInterface,
    NodeInterface,
    TestResult,
    TestResultAction,
    AS_MANY_AS_FIELDS,
    ComparisonSettings,
)


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
        edge_value["correlation"] = correlation(x.values, y.values)
        # edge_value["covariance"] = covariance(x.values, y.values)
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

    def test(self, nodes: List[str], graph: BaseGraphInterface) -> TestResult:
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
        t, critical_t = get_t_and_critial_t(
            sample_size, nb_of_control_vars, corr, self.threshold
        )
        if abs(t) < critical_t:
            return TestResult(
                x=x,
                y=y,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={},
            )

        return


class PartialCorrelationTest(IndependenceTestInterface):
    GENERATOR = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=3, max=3)
    )
    CHUNK_SIZE_PARALLEL_PROCESSING = 1
    PARALLEL = False

    def test(self, nodes: Tuple[str], graph: BaseGraphInterface) -> TestResult:
        """
        Test if nodes x,y are independent given node z based on a partial correlation test.
        We use this test for all combinations of 3 nodes because it is faster than the extended test (which supports combinations of n nodes). We can
        use it to remove edges between nodes which are not independent given another node and so reduce the number of combinations for the extended test.
        :param nodes: the nodes to test
        :return: A TestResult with the action to take
        """
        results = []
        for nodes in itertools.permutations(nodes):
            x: NodeInterface = graph.nodes[nodes[0]]
            y: NodeInterface = graph.nodes[nodes[1]]
            z: NodeInterface = graph.nodes[nodes[2]]

            # Avoid division by zero
            if x is None or y is None or z is None:
                return
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
            t, critical_t = get_t_and_critial_t(
                sample_size, nb_of_control_vars, par_corr, self.threshold
            )

            if abs(t) < critical_t:
                results.append(
                    TestResult(
                        x=x,
                        y=y,
                        action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                        data={"separatedBy": [z]},
                    )
                )
        return results


class ExtendedPartialCorrelationTestLinearRegression(IndependenceTestInterface):
    GENERATOR = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=5, max=AS_MANY_AS_FIELDS)
    )
    CHUNK_SIZE_PARALLEL_PROCESSING = 1
    PARALLEL = False

    def test(self, nodes: List[str], graph: BaseGraphInterface) -> TestResult:
        """
        Test if nodes x,y are independent given Z (set of nodes) based on partial correlation using linear regression and a correlation test on the residuals.
        We use this test for all combinations of more than 3 nodes because it is slower.
        :param nodes: the nodes to test
        :return: A TestResult with the action to take
        """
        n = len(nodes)
        sample_size = len(graph.nodes[nodes[0]].values)
        nodes_set = set([graph.nodes[n] for n in nodes])

        nb_of_control_vars = n - 2
        results = []
        for i in range(n):
            for j in range(i + 1, n):
                x = graph.nodes[nodes[i]]
                y = graph.nodes[nodes[j]]
                exclude_indices = [i, j]
                other_nodes = [
                    graph.nodes[n].values
                    for idx, n in enumerate(nodes)
                    if idx not in exclude_indices
                ]
                par_corr = get_correlation(x, y, other_nodes)
                logger.debug(f"par_corr {par_corr}")
                # make t test for independence of a and y given other nodes
                t, critical_t = get_t_and_critial_t(
                    sample_size, nb_of_control_vars, par_corr, self.threshold
                )

                if abs(t) < critical_t:
                    results.append(
                        TestResult(
                            x=x,
                            y=y,
                            action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                            data={"separatedBy": list(nodes_set - {x, y})},
                        )
                    )

        return results


class ExtendedPartialCorrelationTestMatrix(IndependenceTestInterface):
    GENERATOR = PairsWithNeighboursGenerator(
        comparison_settings=ComparisonSettings(min=4, max=AS_MANY_AS_FIELDS)
    )
    CHUNK_SIZE_PARALLEL_PROCESSING = 100
    PARALLEL = True

    def test(self, nodes: List[str], graph: BaseGraphInterface) -> TestResult:
        """
        Test if nodes x,y are independent given Z (set of nodes) based on partial correlation using the inverted covariance matrix (precision matrix).
        https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
        We use this test for all combinations of more than 3 nodes because it is slower.
        :param nodes: the nodes to test
        :return: A TestResult with the action to take
        """
        if not graph.edge_exists(graph.nodes[nodes[0]], graph.nodes[nodes[1]]):
            return

        other_neighbours = set(graph.edges[graph.nodes[nodes[0]]]) | set(
            graph.edges[graph.nodes[nodes[1]]]
        )
        other_neighbours.remove(graph.nodes[nodes[0]])
        other_neighbours.remove(graph.nodes[nodes[1]])

        if not set(nodes[2:]).issubset(set([on.name for on in list(other_neighbours)])):
            return

        covariance_matrix = [
            [None for _ in range(len(nodes))] for _ in range(len(nodes))
        ]
        for i in range(len(nodes)):
            for k in range(i, len(nodes)):
                if covariance_matrix[i][k] is None:
                    covariance_matrix[i][k] = covariance(
                        graph.nodes[nodes[i]].values, graph.nodes[nodes[k]].values
                    )
                    covariance_matrix[k][i] = covariance_matrix[i][k]

        cov_matrix = np.array(covariance_matrix)
        inverse_cov_matrix = np.linalg.inv(cov_matrix)
        n = len(inverse_cov_matrix)
        diagonal = np.diagonal(inverse_cov_matrix)
        diagonal_matrix = np.zeros((n, n))
        np.fill_diagonal(diagonal_matrix, diagonal)
        helper = np.dot(np.sqrt(diagonal_matrix), inverse_cov_matrix)
        precision_matrix = np.dot(helper, np.sqrt(diagonal_matrix))

        sample_size = len(graph.nodes[nodes[0]].values)
        nb_of_control_vars = len(nodes) - 2
        results = []

        nodes_set = set([graph.nodes[n] for n in nodes])
        deleted_edges = []

        for i in range(len(precision_matrix)):
            for k in range(len(precision_matrix[i])):
                if i == k:
                    continue
                if (nodes[i], nodes[k]) in deleted_edges or (
                    nodes[k],
                    nodes[i],
                ) in deleted_edges:
                    continue
                try:
                    t, critical_t = get_t_and_critial_t(
                        sample_size,
                        nb_of_control_vars,
                        (
                            -precision_matrix[i][k]
                            / math.sqrt(precision_matrix[i][i] * precision_matrix[k][k])
                        ),
                        self.threshold,
                    )
                except ValueError:
                    # TODO: @sof fiugre out why this happens
                    logger.debug(f"ValueError {i} {k} ({precision_matrix[i][k]})")
                    continue

                if abs(t) < critical_t:
                    deleted_edges.append((nodes[i], nodes[k]))
                    results.append(
                        TestResult(
                            x=graph.nodes[nodes[i]],
                            y=graph.nodes[nodes[k]],
                            action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                            data={
                                "separatedBy": list(
                                    nodes_set
                                    - {graph.nodes[nodes[i]], graph.nodes[nodes[k]]}
                                )
                            },
                        )
                    )
        return results


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
