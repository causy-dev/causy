import enum
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

import torch

from causy.serialization import SerializeMixin
from causy.utils import load_pipeline_artefact_by_definition

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.01

AS_MANY_AS_FIELDS = 0


@dataclass
class ComparisonSettings(SerializeMixin):
    min: int = 2
    max: int = AS_MANY_AS_FIELDS


class NodeInterface(SerializeMixin):
    name: str
    id: str
    values: torch.Tensor

    def serialize(self):
        return {"id": self.id, "name": self.name}


class TestResultAction(enum.StrEnum):
    REMOVE_EDGE_UNDIRECTED = "REMOVE_EDGE_UNDIRECTED"
    UPDATE_EDGE = "UPDATE_EDGE"
    UPDATE_EDGE_DIRECTED = "UPDATE_EDGE_DIRECTED"
    DO_NOTHING = "DO_NOTHING"
    REMOVE_EDGE_DIRECTED = "REMOVE_EDGE_DIRECTED"


@dataclass
class TestResult(SerializeMixin):
    x: NodeInterface
    y: NodeInterface
    action: TestResultAction
    data: Optional[Dict] = None

    def serialize(self):
        return {
            "x": self.x.serialize(),
            "y": self.y.serialize(),
            "action": self.action.name,
        }


class BaseGraphInterface(ABC):
    nodes: Dict[str, NodeInterface]
    edges: Dict[str, Dict[str, Dict]]

    @abstractmethod
    def retrieve_edge_history(self, u, v, action: TestResultAction) -> List[TestResult]:
        pass

    @abstractmethod
    def add_edge_history(self, u, v, action: TestResult):
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
    def add_node(self, name, values) -> NodeInterface:
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


class GeneratorInterface(ABC, SerializeMixin):
    comparison_settings: ComparisonSettings
    chunked: bool = False

    @abstractmethod
    def generate(self, graph: BaseGraphInterface, graph_model_instance_: dict):
        pass

    def __init__(self, comparison_settings: ComparisonSettings, chunked: bool = None):
        if isinstance(comparison_settings, dict):
            comparison_settings = load_pipeline_artefact_by_definition(
                comparison_settings
            )

        if chunked is not None:
            self.chunked = chunked

        self.comparison_settings = comparison_settings


class IndependenceTestInterface(ABC, SerializeMixin):
    # TODO: refactor IndependenceTestInterface -> PipelineStepInterface or so
    num_of_comparison_elements: int = 0
    generator: Optional[GeneratorInterface] = None

    chunk_size_parallel_processing: int = 1

    parallel: bool = True

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
                self.generator = load_pipeline_artefact_by_definition(generator)
            else:
                self.generator = generator

        if num_of_comparison_elements:
            if isinstance(num_of_comparison_elements, dict):
                self.num_of_comparison_elements = load_pipeline_artefact_by_definition(
                    num_of_comparison_elements
                )
            else:
                self.num_of_comparison_elements = num_of_comparison_elements

        if chunk_size_parallel_processing:
            self.chunk_size_parallel_processing = chunk_size_parallel_processing

        if parallel:
            self.parallel = parallel

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


class LogicStepInterface(ABC, SerializeMixin):
    @abstractmethod
    def execute(self, graph: BaseGraphInterface, graph_model_instance_: dict):
        pass


class ExitConditionInterface(ABC, SerializeMixin):
    @abstractmethod
    def check(
        self,
        graph: BaseGraphInterface,
        graph_model_instance_: GraphModelInterface,
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
        graph_model_instance_: GraphModelInterface,
        actions_taken: List[TestResult],
        iteration: int,
    ) -> bool:
        return self.check(graph, graph_model_instance_, actions_taken, iteration)
