import abc
import copy
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import random as python_random

from causy.graph import GraphManager
from typing import Dict, Callable, List, Optional, Union

import logging

logger = logging.getLogger(__name__)


def random_normal() -> float:
    """
    Returns a random number from a normal distribution
    :return: the random number as a float
    """
    return python_random.normalvariate(0, 1)


@dataclass
class NodeReference:
    """
    A reference to a node in the sample generator
    """

    node: str

    def __str__(self):
        return self.node


@dataclass
class TimeAwareNodeReference(NodeReference):
    """
    A reference to a node in the sample generator
    """

    point_in_time: int = 0

    def __str__(self):
        return f"{self.node} - t{self.point_in_time}"


@dataclass
class SampleEdge:
    """
    An edge in the sample generator that references a node and a lag
    """

    from_node: NodeReference
    to_node: NodeReference
    value: float = 0


class TimeTravelingError(Exception):
    """
    An error that is raised when a TimeProxy tries to access a value from the future
    """

    pass


class AbstractSampleGenerator(abc.ABC):
    def __init__(
        self,
        edges: List[Union[SampleEdge, SampleEdge]],
        random: Callable = random_normal,  # for setting that to a fixed value for testing use random = lambda: 0
    ):
        self._edges = edges
        self._variables = self._find_variables_in_edges()
        self.random_fn = random

    @abc.abstractmethod
    def generate(self, size: int) -> SimpleNamespace:
        """
        Generate data for a sample graph with a time dimension
        :param size: the number of time steps to generate
        :return: the generated data and the sample graph
        """
        pass

    @abc.abstractmethod
    def _generate_data(self, size: int) -> Dict[str, torch.Tensor]:
        """
        Generate data for a sample graph with a time dimension
        :param size: the number of time steps to generate
        :return: the generated data
        """
        pass

    def _find_variables_in_edges(self):
        """
        Find all variables in the edges of the sample generator. We need this to calculate the initial values for all existing variables.
        :return: a set of all variables submitted via the edges
        """
        variables = set()
        for edge in self._edges:
            variables.add(edge.from_node.node)
            variables.add(edge.to_node.node)
        return variables

    def _get_edges_for_node_to(self, node: str):
        """
        Get all edges that point to a specific node
        :param node: the node to get the edges for
        :return: a list of edges
        """
        return [edge for edge in self._edges if edge.to_node.node == node]


class IIDSampleGenerator(AbstractSampleGenerator):
    """
    A sample generator that generates data from i.i.d multivariate Gaussians.

    A variable can not depend on itself.

    Example:
    >>> sg = IIDSampleGenerator(
    >>>     edges=[
    >>>      SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
    >>>      SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
    >>>      ],
    >>> )

    """

    def __init__(
        self,
        edges: List[Union[SampleEdge, SampleEdge]],
        random: Callable = random_normal,
    ):
        super().__init__(edges, random)
        self._variables = self.topologic_sort(
            sorted(copy.deepcopy(list(self._variables)))
        )

    def topologic_sort(self, nodes: List[str]):
        """
        Sorts the nodes topologically
        :param nodes: list of nodes
        :param edges: list of edges
        :return: a list of sorted nodes
        """
        sorted_nodes = []
        while nodes:
            for node in nodes:
                if set(
                    [edge.from_node.node for edge in self._get_edges_for_node_to(node)]
                ).issubset(set(sorted_nodes)):
                    sorted_nodes.append(node)
                    nodes.remove(node)
                    break
        return sorted_nodes

    def _generate_data(self, size):
        """
        Generate data for a sample graph with a time dimension
        :param size: the number of time steps to generate
        :return: the generated data
        """
        internal_repr = {}

        # Initialize the output dictionary by adding noise
        for k in self._variables:
            internal_repr[k] = torch.tensor(
                [self.random_fn() for _ in range(size)], dtype=torch.float64
            )

        # Iterate over the nodes and sort them by the number of ingoing edges
        ingoing_edges = {}

        # topological sort: sort the nodes in an order that the nodes that depend on other nodes come after the nodes they depend on
        for node_name in self._variables:
            # Get the edges that point to this node
            ingoing_edges[node_name] = self._get_edges_for_node_to(node_name)

        # Sort the node such that all nodes that appear in edges must have occured as keys before
        sorted_nodes = self._variables

        # Generate the data
        for to_node in sorted_nodes:
            for edge in ingoing_edges[to_node]:
                from_node = edge.from_node.node
                internal_repr[to_node] += edge.value * internal_repr[from_node]
        return internal_repr

    # TODO: adjust sample generator tests to this data shape
    def _generate_shaped_data(self, size):
        test_data = self._generate_data(size)
        data = {}
        for variable in test_data.keys():
            data[variable] = test_data[variable]

        test_data_shaped = []
        for i in range(size):
            entry = {}
            for key in data.keys():
                entry[key] = data[key][i]
            test_data_shaped.append(entry)
        return test_data_shaped

    def generate(self, size):
        """
        Generate data for a sample graph with a time dimension
        :param size: the number of time steps to generate
        :return: the generated data and the sample graph
        """
        output = self._generate_data(size)
        graph = GraphManager()
        for i in self._variables:
            graph.add_node(
                i,
                output[i],
                id_=i,
                metadata={"variable": i},
            )

        for edge in self._edges:
            graph.add_directed_edge(
                graph.nodes[edge.from_node.node],
                graph.nodes[edge.to_node.node],
                metadata={},
            )

        return output, graph


# TODO: Does not work for multiple lags properly yet and is numerically unstable for several cases (check why, fix it)
class TimeseriesSampleGenerator(AbstractSampleGenerator):
    """
    A sample generator that generates data for a sample graph with a time dimension.

    Edges are defined as SampleLaggedEdges, which define a directed edge from a source node to a target node with a
    ag. The lag is the number of time steps between the source and the target.

    A variable can depend on itself, but only on its past values (with a lag). This corresponds to autoregressive models.

    Example:
    >>> sg = TimeseriesSampleGenerator(
    >>>     edges=[
    >>>      SampleEdge(TimeAwareNodeReference("X", -1), TimeAwareNodeReference("X"), 0.9),
    >>>      SampleEdge(TimeAwareNodeReference("Y", -1), TimeAwareNodeReference("Y"), 0.9),
    >>>      SampleEdge(TimeAwareNodeReference("Z", -1), TimeAwareNodeReference("Z"), 0.9),
    >>>      SampleEdge(TimeAwareNodeReference("Z", -1), TimeAwareNodeReference("Y"), 5),
    >>>      SampleEdge(TimeAwareNodeReference("Y", -1), TimeAwareNodeReference("X"), 7),
    >>>      ],
    >>> )
    """

    def __init__(
        self,
        edges: List[Union[SampleEdge, SampleEdge]],
        random: Callable = random_normal,  # for setting that to a fixed value for testing use random = lambda: 0
    ):
        super().__init__(edges, random)
        self._longest_lag = max(
            [abs(edge.from_node.point_in_time) for edge in self._edges]
        )

    _initial_distribution_fn: Callable = lambda self, x: torch.normal(0, x)

    def _generate_data(self, size):
        """
        Generate data for a sample graph with a time dimension
        :param size: the number of time steps to generate
        :return: the generated data
        """
        internal_repr = {}

        initial_values = self._calculate_initial_values()
        # Initialize the output dictionary
        for k in self._variables:
            internal_repr[k] = [initial_values[k]]

        for t in range(1, size):
            for node_name in self._variables:
                # Get the edges that point to this node
                edges = self._get_edges_for_node_to(node_name)
                result = torch.tensor(0.0, dtype=torch.float64)
                for edge in edges:
                    if abs(edge.from_node.point_in_time) > t:
                        result += (
                            edge.value * initial_values[edge.from_node.node]
                        )  # TODO(sofia): map here to the proper point in time
                    else:
                        result += (
                            edge.value
                            * internal_repr[edge.from_node.node][
                                t + edge.from_node.point_in_time
                            ]
                        )

                result += self.random_fn()
                internal_repr[node_name].append(result)

        for k, v in internal_repr.items():
            internal_repr[k] = torch.stack(v)

        return internal_repr

    def generate(self, size):
        """
        Generate data for a sample graph with a time dimension
        :param size: the number of time steps to generate
        :return: the generated data and the sample graph
        """
        output = self._generate_data(size)
        graph = GraphManager()
        for i in self._variables:
            for t in range(size):
                graph.add_node(
                    f"{i} - t{t}",
                    [output[i][t]],
                    id_=f"{i}-t{t}",
                    metadata={"time": t, "variable": i},
                )

        for t in range(1, size):
            for edge in self._edges:
                if t - abs(edge.from_node.point_in_time) < 0:
                    logger.debug(
                        f"Cannot generate data for {edge.from_node.node} at t={t}, "
                        f"since it depends on {abs(edge.to_node.point_in_time)}-steps-ago value"
                    )
                else:
                    graph.add_directed_edge(
                        graph.nodes[
                            f"{edge.from_node.node}-t{t - abs(edge.from_node.point_in_time)}"
                        ],
                        graph.nodes[f"{edge.to_node.node}-t{t}"],
                        metadata={},
                    )

        return output, graph

    def custom_block_diagonal(self, matrices):
        # Get dimensions
        num_matrices = len(matrices)
        n = len(matrices[0])  # Assuming all matrices are of the same size

        # Compute total size of the block diagonal matrix
        total_rows = num_matrices * n
        total_cols = num_matrices * n

        # Create an empty tensor to hold the block diagonal matrix
        block_diag = torch.zeros(total_rows, total_cols)

        # Fill in the first rows with the input matrices
        for i, matrix in enumerate(matrices):
            block_diag[:n, i * n : (i + 1) * n] = torch.tensor(
                matrix, dtype=torch.float64
            )

        # Fill in the lower left off-diagonal with identity matrices
        row_start = n
        col_start = 0
        for i in range(1, num_matrices):
            block_diag[
                row_start : row_start + n, col_start : col_start + n
            ] = torch.eye(n)
            row_start += n
            col_start += n

        return block_diag

    def __generate_coefficient_matrix(self):
        """
        generate the coefficient matrix for the sample generator graph
        :return: the coefficient matrix
        """

        matrix: List[List[List[float]]] = [
            [[0 for _ in self._variables] for _ in self._variables]
            for _ in range(self._longest_lag)
        ]

        # map the initial values to numbers from 0 to n
        values_map = self.__matrix_position_mapping()

        for edge in self._edges:
            matrix[(edge.from_node.point_in_time * -1) - 1][
                values_map[edge.to_node.node]
            ][values_map[edge.from_node.node]] = edge.value

        # return me as torch tensor
        return self.custom_block_diagonal(matrix)

    def vectorize_identity_block(self, n):
        # Create an empty tensor
        matrix = torch.zeros(n, n)

        # Fill the upper left block with an identity matrix
        matrix[: n // self._longest_lag, : n // self._longest_lag] = torch.eye(
            n // self._longest_lag
        )

        # Flatten the matrix
        vectorized_matrix = matrix.view(-1)

        return vectorized_matrix

    def _calculate_initial_values(self):
        """
        Calculate the initial values for the sample generator graph using the covariance matrix of the noise terms.

        coefficient_matrix=[[a,0],[b,a]], i.e.
        coefficient_matrix[0][0] is the coefficient of X_t-1 in the equation for X_t, here a
        coefficient_matrix[0][1] is the coefficient of Y_t-1 in the equation for X_t (here: no edge, that means zero)
        coefficient_matrix[1][1] is the coefficient of Y_t-1 in the equation for Y_t, here a
        coefficient_matrix[1][0] is the coefficient of X_t-1 in the equation for Y_t, here b

        If the top left entry ([0][0]) in our coefficient matrix corresponds to X_{t-1} -> X_t, then the the top left
        entry in the covariance matrix is the variance of X_t, i.e. V(X_t).

        If the top right entry ([0][1]) in our coefficient matrix corresponds to X_{t-1} -> Y_t, then the the top left
        entry in the covariance matrix is the variance of X_t, i.e. V(X_t).
        :return: the initial values for the sample generator graph as a dictionary
        """
        coefficient_matrix = self.__generate_coefficient_matrix()
        n, _ = coefficient_matrix.shape

        kronecker_product = torch.kron(coefficient_matrix, coefficient_matrix)
        identity_matrix = torch.eye(n**2)
        cov_matrix_noise_terms_vectorized = self.vectorize_identity_block(n)
        inv_identity_minus_kronecker_product = torch.linalg.pinv(
            identity_matrix - kronecker_product
        )
        vectorized_covariance_matrix = torch.matmul(
            inv_identity_minus_kronecker_product,
            cov_matrix_noise_terms_vectorized,
        )
        vectorized_covariance_matrix = vectorized_covariance_matrix.reshape(n, n)

        initial_values: Dict[str, torch.Tensor] = {}
        values = torch.diagonal(vectorized_covariance_matrix, offset=0)
        for k, i in self.__matrix_position_mapping().items():
            initial_values[k] = self._initial_distribution_fn(torch.sqrt(values[i]))
        return initial_values

    def __matrix_position_mapping(self):
        """
        Map the variables to numbers from 0 to n. This is needed to calculate the initial values for the sample generator graph.
        :return:
        """
        values_map = {}
        for i, k in enumerate(self._variables):
            values_map[k] = i
        return values_map
