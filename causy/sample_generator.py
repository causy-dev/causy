import abc
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from causy.graph import Graph
from typing import Dict, Callable, List, Optional, Union

import logging

logger = logging.getLogger(__name__)


def random() -> torch.Tensor:
    """
    Returns a random number from a normal distribution
    :return: the random number as a float
    """
    return torch.randn(1, dtype=torch.float32).item()


class SampleEdge:
    """
    Represents an edge in a sample graph
    defines a directed edge from source to target

    Example:
    SampleEdge("X", "Y") defines an edge from X to Y

    """

    def __init__(self, source, target):
        self.source = source
        self.target = target


class SampleLaggedEdge(SampleEdge):
    """
    Represents a lagged edge in a time series sample graph.
    Defines a directed edge from source to target with a lag of lag. The lag is the number of time steps between the
    source and the target.

    Example:
    SampleLaggedEdge("X", "Y", 1) defines an edge from X to Y with a lag of 1

    """

    def __init__(self, source, target, lag):
        super().__init__(source, target)
        self.lag = lag


class TimeTravelingError(Exception):
    """
    An error that is raised when a TimeProxy tries to access a value from the future
    """

    pass


class TimeProxy:
    """
    A proxy object that allows to access past values of a variable by using the t() method.

    Example:
    >>> tp = TimeProxy(5)
    >>> tp.set_current_time(1)
    >>> tp.t(-1)
    5
    """

    def __init__(self, initial_value: torch.Tensor):
        """
        :param initial_value: the initial value of the variable
        """
        if not isinstance(initial_value, torch.Tensor):
            initial_value = torch.tensor(initial_value, dtype=torch.float32)
        self._lst = [initial_value]
        self._t = 0

    def set_current_time(self, t):
        """
        Set the current time step which t will be relative to
        :param t: time step as an integer
        :return:
        """
        self._t = t

    def t(self, t):
        """
        Return the value of the variable at time step t
        :param t: the relative time step as a negative integer
        :return: the value of the variable at time step t or a random number if t is too large
        :raises TimeTravelingError: if t is positive
        """
        if t > 0:
            raise TimeTravelingError(
                f"Cannot travel into the future ({self._t + t} steps ahead)."
            )
        elif t + self._t < 0:
            # TODO this might be a bad idea
            return random()
        return self._lst[self._t + t]

    def append(self, value):
        """
        Append a value to the list of values
        :param value: the value to append
        :return:
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        self._lst.append(value)

    def to_list(self):
        """
        Return the list of values
        :return: the list
        """
        return torch.stack(self._lst)

    def __repr__(self):
        return f"{self._lst} at {self._t}"


class CurrentElementProxy(float):
    """
    A proxy object that allows to access the current value of a variable. It is a subclass of float, so it can be used
    as a float.
    """

    # TODO: fix this class. Bug: IIDSampleGenerator only depends on the initial value, it should depend on the step

    def __init__(self, initial_value: torch.Tensor):
        if not isinstance(initial_value, torch.Tensor):
            initial_value = torch.tensor(initial_value, dtype=torch.float32)
        self.lst = [initial_value]
        self.value = initial_value

    def append(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)

        self.lst.append(value)
        self.value = value

    def to_list(self):
        return torch.stack(self.lst)


class AbstractSampleGenerator(abc.ABC):
    """
    An abstract class for sample generators that generate data for a sample graph.
    It is implemented by TimeseriesSampleGenerator and IIDSampleGenerator.

    """

    def __init__(
        self,
        initial_values: Dict[str, any],
        generators: Dict[str, Callable],
        edges: List[Union[SampleLaggedEdge, SampleEdge]],
        variables: Optional[Dict[str, any]] = None,
    ):
        self.initial_values = initial_values
        self.generators = generators
        self.edges = edges
        if variables is None:
            variables = {}
        self.vars = variables

    @abc.abstractmethod
    def generate(self, size: int):
        pass


class IIDSampleGenerator(AbstractSampleGenerator):
    """
    A sample generator that generates data for a sample graph without a time dimension.

    Generators are written as a lambda function that takes two arguments: the current step and the input.
    The input is a SimpleNamespace object that contains the current values of all variables.

    A variable can not depend on itself.


    Example:
    >>> sg = IIDSampleGenerator(
    >>>     initial_values={
    >>>         "Z": random(),
    >>>         "Y": random(),
    >>>         "X": random(),
    >>>     },
    >>>     variables={
    >>>         "param1": 2,
    >>>         "param2": 3,
    >>>     },
    >>>     # generate the dependencies of variables on values of themselves and other variables
    >>>     generators={
    >>>         "Z": lambda s, i: random(),
    >>>         "Y": lambda s, i: i.param1 * i.Z + random(),
    >>>         "X": lambda s, i: i.param2 * i.Y + random()
    >>>     },
    >>>     edges=[
    >>>         SampleEdge("Z", "Y"),
    >>>         SampleEdge("Y", "X"),
    >>>     ]
    >>> )

    """

    # TODO: fix this class. Bug: IIDSampleGenerator only depends on the initial value, it should depend on the step
    def _generate_data(self, size):
        internal_repr = {}

        # Initialize the output dictionary
        for i, v in self.initial_values.items():
            internal_repr[i] = CurrentElementProxy(v)

        # Generate the data for each time step
        for step in range(1, size):
            for value in self.generators.keys():
                generator_input = internal_repr
                generator_input.update(self.vars)
                internal_repr[value].append(
                    self.generators[value](step, SimpleNamespace(**generator_input))
                )
        output = {}
        for i in self.generators.keys():
            output[i] = internal_repr[i].to_list()

        return output

    def generate(self, size):
        """
        Generate data for a sample graph without a time dimension
        :param size: the number of data points to generate
        :return: the generated data and the sample graph
        """
        output = self._generate_data(size)
        graph = Graph()

        for i in self.generators.keys():
            graph.add_node(f"{i}", output[i], id_=f"{i}")

        for edge in self.edges:
            graph.add_edge(
                graph.nodes[f"{edge.source}"],
                graph.nodes[f"{edge.target}"],
                metadata={},
            )
            graph.remove_directed_edge(
                graph.nodes[f"{edge.target}"], graph.nodes[f"{edge.source}"]
            )

        return output, graph


@dataclass
class NodeReference:
    """
    A reference to a node in the sample generator
    """

    node: str
    point_in_time: int = 0


@dataclass
class SampleLaggedEdge:
    """
    An edge in the sample generator that references a node and a lag
    """

    from_node: NodeReference
    to_node: NodeReference
    value: float = 0


class TimeseriesSampleGenerator:
    def __init__(
        self,
        edges: List[Union[SampleLaggedEdge, SampleLaggedEdge]],
        random: Callable = random,  # for setting that to a fixed value for testing use random = lambda: 0
    ):
        self.__edges = edges
        self.__variables = self.__find__variables_in_edges()
        self.__longest_lag = max(
            [abs(edge.to_node.point_in_time) for edge in self.__edges]
        )
        self.random_fn = random

    def __find__variables_in_edges(self):
        variables = set()
        for edge in self.__edges:
            variables.add(edge.from_node.node)
            variables.add(edge.to_node.node)
        return variables

    random_fn: Callable = random
    _initial_distribution_fn: Callable = lambda self, x: torch.normal(0, x)

    def get_edges_for_node_to(self, node: str):
        return [edge for edge in self.__edges if edge.to_node.node == node]

    def _generate_data(self, size):
        """
        Generate data for a sample graph with a time dimension
        :param size:
        :return:
        """
        internal_repr = {}

        initial_values = self._calculate_initial_values()

        # Initialize the output dictionary
        for k in self.__variables:
            internal_repr[k] = [initial_values[k]]

        for t in range(1, size):
            for node_name in self.__variables:
                # Get the edges that point to this node
                edges = self.get_edges_for_node_to(node_name)
                result = torch.tensor(0.0, dtype=torch.float32)
                for edge in edges:
                    if abs(edge.to_node.point_in_time) > t:
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
        graph = Graph()
        for i in self.__variables:
            for t in range(size):
                graph.add_node(
                    f"{i} - t{t}",
                    [output[i][t]],
                    id_=f"{i}-t{t}",
                    metadata={"time": t, "variable": i},
                )

        for t in range(1, size):
            for edge in self.__edges:
                if t - abs(edge.to_node.point_in_time) < 0:
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

    def __generate_coefficient_matrix(self):
        """
        generate the coefficient matrix for the sample generator graph
        :return:
        """

        matrix: List[List[float]] = [
            [0 for _ in self.__variables] for _ in self.__variables
        ]

        # map the initial values to numbers from 0 to n
        values_map = self.__matrix_position_mapping()
        for i, k in enumerate(self.__variables):
            values_map[k] = i

        for edge in self.__edges:
            matrix[values_map[edge.to_node.node]][
                values_map[edge.from_node.node]
            ] = edge.value
        # return me as torch tensor
        return matrix

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
        """
        coefficient_matrix = torch.tensor(
            self.__generate_coefficient_matrix(), dtype=torch.float32
        )

        kronecker_product = torch.kron(coefficient_matrix, coefficient_matrix)
        n, _ = coefficient_matrix.shape
        identity_matrix = torch.eye(n**2)
        cov_matrix_noise_terms_vectorized = torch.eye(n).flatten()
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
        for i, k in enumerate(self.__variables):
            initial_values[k] = self._initial_distribution_fn(torch.sqrt(values[i]))

        return initial_values

    def __matrix_position_mapping(self):
        values_map = {}
        for i, k in enumerate(self.__variables):
            values_map[k] = i
        return values_map
