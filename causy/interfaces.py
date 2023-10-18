import enum
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.01

AS_MANY_AS_FIELDS = 0


@dataclass
class ComparisonSettings:
    min: int = 2
    max: int = AS_MANY_AS_FIELDS


class NodeInterface:
    name: str
    values: List[float]

    def to_dict(self):
        return self.name


class TestResultAction(enum.StrEnum):
    REMOVE_EDGE_UNDIRECTED = "REMOVE_EDGE_UNDIRECTED"
    UPDATE_EDGE = "UPDATE_EDGE"
    DO_NOTHING = "DO_NOTHING"
    REMOVE_EDGE_DIRECTED = "REMOVE_EDGE_DIRECTED"


@dataclass
class TestResult:
    x: NodeInterface
    y: NodeInterface
    action: TestResultAction
    data: Dict = None

    def to_dict(self):
        return {
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
            "action": self.action.name,
        }


class BaseGraphInterface(ABC):
    nodes: Dict[str, NodeInterface]
    edges: Dict[NodeInterface, Dict[NodeInterface, Dict]]

    @abstractmethod
    def retrieve_edge_history(self, u, v, action: TestResultAction) -> List[TestResult]:
        pass

    @abstractmethod
    def add_edge_history(self, u, v, action: TestResultAction):
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
    def undirected_edge_exists(self, u, v):
        pass

    @abstractmethod
    def directed_edge_exists(self, u, v):
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
    GENERATOR = None

    CHUNK_SIZE_PARALLEL_PROCESSING = 1

    PARALLEL = True

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    @abstractmethod
    def test(self, nodes: List[str], graph: BaseGraphInterface) -> TestResult:
        """
        Test if x and y are independent
        :param x: x values
        :param y: y values
        :return: True if independent, False otherwise
        """
        pass

    def __call__(self, nodes: List[str], graph: BaseGraphInterface) -> TestResult:
        return self.test(nodes, graph)


class LogicStepInterface(ABC):
    @abstractmethod
    def execute(self, graph: BaseGraphInterface, graph_model_instance_: dict):
        pass


class GeneratorInterface(ABC):
    @abstractmethod
    def generate(self, graph: BaseGraphInterface, graph_model_instance_: dict):
        pass


class ExitConditionInterface(ABC):
    @abstractmethod
    def check(
        self,
        graph: BaseGraphInterface,
        graph_model_instance_: dict,
        actions_taken: List[TestResult],
        iteration: int,
    ) -> bool:
        """
        :param graph:
        :param graph_model_instance_:
        :param actions_taken:
        :param iteration:
        :return: True if you want to break an iteration, False otherwise
        """
        pass

    def __call__(
        self,
        graph: BaseGraphInterface,
        graph_model_instance_: dict,
        actions_taken: List[TestResult],
        iteration: int,
    ) -> bool:
        return self.check(graph, graph_model_instance_, actions_taken, iteration)
