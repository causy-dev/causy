import enum
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

DEFAULT_THRESHOLD = 0.01

AS_MANY_AS_FIELDS = 0


@dataclass
class ComparisonSettings:
    min: int = 2
    max: int = AS_MANY_AS_FIELDS


class NodeInterface:
    name: str
    values: List[float]


class CorrelationTestResultAction(enum.Enum):
    REMOVE_EDGE_UNDIRECTED = 1
    UPDATE_EDGE = 2
    DO_NOTHING = 3
    REMOVE_EDGE_DIRECTED = 4


@dataclass
class CorrelationTestResult:
    x: NodeInterface
    y: NodeInterface
    action: CorrelationTestResultAction
    data: Dict = None


class BaseGraphInterface(ABC):
    nodes: Dict[str, NodeInterface]
    edges: Dict[NodeInterface, Dict[NodeInterface, Dict]]

    @abstractmethod
    def retrieve_edge_history(
        self, u, v, action: CorrelationTestResultAction
    ) -> List[CorrelationTestResult]:
        pass

    @abstractmethod
    def add_edge_history(self, u, v, action: CorrelationTestResultAction):
        pass

    @abstractmethod
    def add_edge(self, u, v, w):
        pass

    @abstractmethod
    def remove_edge(self, u, v):
        pass

    @abstractmethod
    def remove_directed_edge(self, u, v):
        pass

    @abstractmethod
    def update_edge(self, u, v, w):
        pass

    @abstractmethod
    def add_node(self, name, values):
        pass

    @abstractmethod
    def edge_value(self, u, v):
        pass

    @abstractmethod
    def edge_exists(self, u, v):
        pass


class GraphModelInterface(ABC):
    pool: multiprocessing.Pool

    @abstractmethod
    def create_graph_from_data(self, data: List[Dict]):
        pass

    @abstractmethod
    def execute_pipeline_steps(self):
        pass

    @abstractmethod
    def execute_pipeline_step(self, step):
        pass


class IndependenceTestInterface(ABC):
    NUM_OF_COMPARISON_ELEMENTS = 0

    CHUNK_SIZE_PARALLEL_PROCESSING = 1

    PARALLEL = True

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    @abstractmethod
    def test(
        self, nodes: List[str], graph: BaseGraphInterface
    ) -> CorrelationTestResult:
        """
        Test if x and y are independent
        :param x: x values
        :param y: y values
        :return: True if independent, False otherwise
        """
        pass

    def __call__(
        self, nodes: List[str], graph: BaseGraphInterface
    ) -> CorrelationTestResult:
        return self.test(nodes, graph)


class LogicStepInterface(ABC):
    @abstractmethod
    def execute(self, graph: BaseGraphInterface, graph_model_instance_: dict):
        pass
