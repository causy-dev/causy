import itertools
from copy import deepcopy
from typing import Tuple, List, Optional
import logging

import torch

from causy.generators import (
    AllCombinationsGenerator,
    PairsWithNeighboursGenerator,
    BatchGenerator,
)
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

from scipy import stats as scipy_stats


class CalculateCorrelations(IndependenceTestInterface):
    generator = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    chunk_size_parallel_processing = 1
    parallel = False

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
        )
        return TestResult(
            x=x,
            y=y,
            action=TestResultAction.UPDATE_EDGE,
            data=edge_value,
        )


class CorrelationCoefficientTest(IndependenceTestInterface):
    generator = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    chunk_size_parallel_processing = 1
    parallel = False

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
    generator = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=3, max=3)
    )
    chunk_size_parallel_processing = 1
    parallel = False

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
            except (KeyError, TypeError):
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
    generator = PairsWithNeighboursGenerator(
        comparison_settings=ComparisonSettings(min=4, max=AS_MANY_AS_FIELDS)
    )
    chunk_size_parallel_processing = 1000
    parallel = False

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
        diagonal_matrix = torch.zeros((n, n), dtype=torch.float32)
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
            ),
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
    num_of_comparison_elements = 2
    chunk_size_parallel_processing = 10
    parallel = False

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


class ExtendedPartialCorrelationTestMatrixWithTorchBatching(IndependenceTestInterface):
    generator = None
    chunk_size_parallel_processing = 1
    parallel = False
    current_device = "cpu"

    def test(self, nodes: List[str], graph: BaseGraphInterface) -> Optional[TestResult]:
        """
        Test if nodes x,y are independent given Z (set of nodes) based on partial correlation using the inverted covariance matrix (precision matrix).
        https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion

        This version uses torch batching to speed up the calculation.

        :param nodes: the nodes to test
        :return: A TestResult with the action to take
        """
        mapped_nodes = {}

        test_results = []

        for node in set(itertools.chain.from_iterable(nodes)):
            mapped_nodes[node] = graph.nodes[node].values.to(self.current_device)

        combinations = []
        local_edges = deepcopy(graph.edges)
        local_nodes = deepcopy(graph.nodes)
        for combination in nodes:
            if not graph.edge_exists(
                local_nodes[combination[0]], local_nodes[combination[1]]
            ):
                continue

            other_neighbours = set(local_edges[combination[0]])
            other_neighbours.remove(local_nodes[combination[1]].id)

            if not set(combination[2:]).issubset(
                set([on for on in list(other_neighbours)])
            ):
                continue
            combinations.append(
                torch.stack([mapped_nodes[node] for node in combination])
            )

        if len(combinations) == 0:
            return
        node_combinations = combinations
        stacked_matrices = torch.stack(node_combinations)

        calculated_matrices = torch.vmap(torch.cov)(stacked_matrices)
        inverted_matrices = torch.vmap(torch.inverse)(calculated_matrices)
        diagonal = torch.vmap(torch.diag)(inverted_matrices)

        diagonal_matrix = torch.vmap(lambda x: torch.diag(x))(diagonal)
        diagonal_matrix = torch.vmap(torch.sqrt)(diagonal_matrix)

        helper = torch.vmap(torch.mm)(diagonal_matrix, inverted_matrices)
        precision_matrix = torch.vmap(torch.mm)(helper, diagonal_matrix)

        del diagonal_matrix
        del inverted_matrices

        parr_corrs = torch.vmap(
            lambda x: x[0][1].mul(-1).div(torch.sqrt(x[0][0].mul(x[1][1])))
        )(precision_matrix)

        del precision_matrix

        sample_size = graph.nodes[nodes[0][0]].values.size(0)
        nb_of_control_vars = len(nodes[0]) - 2
        deg_of_freedom = sample_size - 2 - nb_of_control_vars
        critical_t = torch.tensor(
            float(scipy_stats.t.ppf(1 - self.threshold / 2, deg_of_freedom)),
            dtype=torch.float16,
            device=self.current_device,
        )
        deg_of_freedom = torch.tensor(
            deg_of_freedom, dtype=torch.float16, device=self.current_device
        )

        ts = torch.vmap(
            lambda x: torch.abs(
                x.mul(torch.sqrt(deg_of_freedom / (1 - torch.pow(x, 2))))
            )
        )(parr_corrs)

        del parr_corrs

        indices_to_remove = torch.where(ts < critical_t)[0]

        for i in indices_to_remove:
            nodes_set = set([graph.nodes[n] for n in nodes[i]])
            other_neighbours = nodes_set - {
                graph.nodes[nodes[i][0]],
                graph.nodes[nodes[i][1]],
            }
            logger.debug(
                f"Nodes {graph.nodes[nodes[i][0]].name} and {graph.nodes[nodes[i][1]].name} are uncorrelated given nodes {','.join([on.name for on in other_neighbours])}"
            )

            test_results.append(
                TestResult(
                    x=graph.nodes[nodes[i][0]],
                    y=graph.nodes[nodes[i][1]],
                    action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                    data={"separatedBy": list(other_neighbours)},
                )
            )
        return test_results
