import enum
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

from causy.utils import serialize_module_name, load_pipeline_artefact_by_definition

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.01

AS_MANY_AS_FIELDS = 0


@dataclass
class ComparisonSettings:
    min: int = 2
    max: int = AS_MANY_AS_FIELDS

    def serialize(self):
        return {
            "name": serialize_module_name(self),
            "params": {
                "min": self.min,
                "max": self.max,
            },
        }


class NodeInterface:
    name: str
    id: str
    values: List[float]

    def to_dict(self):
        return {"id": self.id, "name": self.name}


class TestResultAction(enum.StrEnum):
    REMOVE_EDGE_UNDIRECTED = "REMOVE_EDGE_UNDIRECTED"
    UPDATE_EDGE = "UPDATE_EDGE"
    UPDATE_EDGE_DIRECTED = "UPDATE_EDGE_DIRECTED"
    DO_NOTHING = "DO_NOTHING"
    REMOVE_EDGE_DIRECTED = "REMOVE_EDGE_DIRECTED"


@dataclass
class TestResult:
    x: NodeInterface
    y: NodeInterface
    action: TestResultAction
    data: Optional[Dict] = None

    def to_dict(self):
        return {
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
            "action": self.action.name,
        }


class BaseGraphInterface(ABC):
    nodes: Dict[str, NodeInterface]
    edges: Dict[str, Dict[str, Dict]]

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


class GeneratorInterface(ABC):
    comparison_settings: ComparisonSettings
    chunked: bool = False

    @abstractmethod
    def generate(self, graph: BaseGraphInterface, graph_model_instance_: dict):
        pass

    def serialize(self) -> dict:
        return {
            "name": serialize_module_name(self),
            "params": {
                "comparison_settings": self.comparison_settings.serialize()
                if self.comparison_settings
                else None,
                "chunked": self.chunked,
            },
        }

    def __init__(self, comparison_settings: ComparisonSettings, chunked: bool = None):
        if isinstance(comparison_settings, dict):
            comparison_settings = load_pipeline_artefact_by_definition(
                comparison_settings
            )

        if chunked is not None:
            self.chunked = chunked

        self.comparison_settings = comparison_settings


class IndependenceTestInterface(ABC):
    NUM_OF_COMPARISON_ELEMENTS: int = 0
    GENERATOR: Optional[GeneratorInterface] = None

    CHUNK_SIZE_PARALLEL_PROCESSING: int = 1

    PARALLEL: bool = True

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        generator: Optional[GeneratorInterface] = None,
        num_of_comparison_elements: int = None,
        chunk_size_parallel_processing: int = None,
        parallel: bool = None,
    ):
        if generator:
            if isinstance(generator, dict):
                self.GENERATOR = load_pipeline_artefact_by_definition(generator)
            else:
                self.GENERATOR = generator

        if num_of_comparison_elements:
            if isinstance(num_of_comparison_elements, dict):
                self.NUM_OF_COMPARISON_ELEMENTS = load_pipeline_artefact_by_definition(
                    num_of_comparison_elements
                )
            else:
                self.NUM_OF_COMPARISON_ELEMENTS = num_of_comparison_elements

        if chunk_size_parallel_processing:
            self.CHUNK_SIZE_PARALLEL_PROCESSING = chunk_size_parallel_processing

        if parallel:
            self.PARALLEL = parallel

        self.threshold = threshold

    @abstractmethod
    def test(self, nodes: List[str], graph: BaseGraphInterface) -> Optional[TestResult]:
        """
        Test if x and y are independent
        :param x: x values
        :param y: y values
        :return: True if independent, False otherwise
        """
        pass

    def __call__(
        self, nodes: List[str], graph: BaseGraphInterface
    ) -> Optional[TestResult]:
        return self.test(nodes, graph)

    def serialize(self) -> dict:
        return {
            "name": serialize_module_name(self),
            "params": {
                "threshold": self.threshold,
                "generator": self.GENERATOR.serialize(),
                "num_of_comparison_elements": self.NUM_OF_COMPARISON_ELEMENTS,
                "chunk_size_parallel_processing": self.CHUNK_SIZE_PARALLEL_PROCESSING,
                "parallel": self.PARALLEL,
            },
        }


class LogicStepInterface(ABC):
    @abstractmethod
    def execute(self, graph: BaseGraphInterface, graph_model_instance_: dict):
        pass

    def serialize(self) -> dict:
        return {
            "name": serialize_module_name(self),
        }


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

    def serialize(self):
        return {
            "name": serialize_module_name(self),
        }
