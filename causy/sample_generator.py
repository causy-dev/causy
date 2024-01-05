import abc
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


class TimeseriesSampleGenerator(AbstractSampleGenerator):
    """
    A sample generator that generates data for a sample graph with a time dimension.

    Generator functions are written as a lambda function that takes two arguments: the current timestep and the input.
    The input is a SimpleNamespace object that contains the current values of all variables. Via the t(-n) method of the
    TimeProxy class, the generator can access the values of the variables at previous time steps.

    During generation, the generator functions are called in the order of the keys of the generators dictionary. This
    means that the order of the keys determines the order in which the variables are generated. If a variable depends on
    another variable, the generator function of the dependent variable should be defined after the generator function of
    the variable it depends on.

    A variable can depend on itself, but only on its past values (with a lag). This is useful for autoregressive models.

    Example:
    >>> sg = TimeseriesSampleGenerator(
    >>>     initial_values={
    >>>         "Z": random(),
    >>>         "Y": random(),
    >>>         "X": random(),
    >>>     },
    >>>     variables={
    >>>         "alpha": 0.9,
    >>>     },
    >>>     # generate the dependencies of variables on past values of themselves and other variables
    >>>     generators={
    >>>         "Z": lambda t, i: i.Z.t(-1) * i.alpha + random(),
    >>>         "Y": lambda t, i: i.Y.t(-1) * i.alpha + i.Z.t(-1) + random(),
    >>>         "X": lambda t, i: i.X.t(-1) * i.alpha + i.Y.t(-1) + random()
    >>>     },
    >>>     edges=[
    >>>         SampleLaggedEdge("X", "X", 1),
    >>>         SampleLaggedEdge("Y", "Y", 1),
    >>>         SampleLaggedEdge("Z", "Z", 1),
    >>>         SampleLaggedEdge("Y", "Z", 1),
    >>>         SampleLaggedEdge("Y", "X", 4),
    >>>     ]
    >>> )
    """

    def _generate_data(self, size):
        """
        Generate data for a sample graph with a time dimension
        :param size:
        :return:
        """
        internal_repr = {}

        # Initialize the output dictionary
        for i, v in self.initial_values.items():
            internal_repr[i] = TimeProxy(v)

        # Generate the data for each time step
        for t in range(1, size):
            for value in self.generators.keys():
                internal_repr[value].set_current_time(t)
                generator_input = internal_repr
                generator_input.update(self.vars)
                internal_repr[value].append(
                    self.generators[value](t, SimpleNamespace(**generator_input))
                )

        output = {}
        for i in self.generators.keys():
            output[i] = internal_repr[i].to_list()

        return output

    def generate(self, size):
        """
        Generate data for a sample graph with a time dimension
        :param size: the number of time steps to generate
        :return: the generated data and the sample graph
        """
        output = self._generate_data(size)
        graph = Graph()
        for i in self.generators.keys():
            for t in range(size):
                graph.add_node(
                    f"{i} - t{t}",
                    [output[i][t]],
                    id_=f"{i}-t{t}",
                    metadata={"time": t, "variable": i},
                )

        for t in range(1, size):
            for edge in self.edges:
                if t - edge.lag < 0:
                    logger.debug(
                        f"Cannot generate data for {edge.source} at t={t}, "
                        f"since it depends on {edge.lag}-steps-ago value"
                    )
                else:
                    graph.add_edge(
                        graph.nodes[f"{edge.source}-t{t - edge.lag}"],
                        graph.nodes[f"{edge.target}-t{t}"],
                        metadata={},
                    )
                    graph.remove_directed_edge(
                        graph.nodes[f"{edge.target}-t{t}"],
                        graph.nodes[f"{edge.source}-t{t - edge.lag}"],
                    )

        return output, graph


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
                graph.nodes[f"{edge.source}"], graph.nodes[f"{edge.target}"], value={}
            )
            graph.remove_directed_edge(
                graph.nodes[f"{edge.target}"], graph.nodes[f"{edge.source}"]
            )

        return output, graph
