import enum
import multiprocessing
from abc import ABC, abstractmethod
from datetime import datetime

from pydantic.dataclasses import dataclass
from typing import List, Dict, Optional, Union, TypeVar, Generic, Any
import logging

import torch

from causy.graph_utils import (
    load_pipeline_artefact_by_definition,
    serialize_module_name,
    load_pipeline_steps_by_definition,
)

from pydantic import BaseModel, computed_field, AwareDatetime, Field

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.01

AS_MANY_AS_FIELDS = 0


class ComparisonSettings(BaseModel):
    min: int = 2
    max: int = AS_MANY_AS_FIELDS

    @computed_field
    @property
    def name(self) -> str:
        return serialize_module_name(self)


class NodeInterface(BaseModel):
    """
    Node interface for the graph. A node is defined by a name and a value.
    """

    name: str
    id: str
    values: Optional[torch.DoubleTensor] = Field(exclude=True, default=None)

    class Config:
        arbitrary_types_allowed = True


class EdgeTypeInterface(BaseModel):
    """
    Edge type interface for the graph
    An edge type is defined by a name
    """

    name: str

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class EdgeInterface(BaseModel):
    """
    Edge interface for the graph
    A graph edge is defined by two nodes and an edge type. It can also have metadata.
    """

    u: NodeInterface
    v: NodeInterface
    edge_type: EdgeTypeInterface
    metadata: Dict[str, any] = None

    class Config:
        arbitrary_types_allowed = True


class TestResultAction(enum.StrEnum):
    """
    Actions that can be taken on the graph. These actions are used to keep track of the history of the graph.
    """

    REMOVE_EDGE_UNDIRECTED = "REMOVE_EDGE_UNDIRECTED"
    UPDATE_EDGE = "UPDATE_EDGE"
    UPDATE_EDGE_TYPE = "UPDATE_EDGE_TYPE"
    UPDATE_EDGE_DIRECTED = "UPDATE_EDGE_DIRECTED"
    UPDATE_EDGE_TYPE_DIRECTED = "UPDATE_EDGE_TYPE_DIRECTED"
    DO_NOTHING = "DO_NOTHING"
    REMOVE_EDGE_DIRECTED = "REMOVE_EDGE_DIRECTED"


class TestResult(BaseModel):
    u: NodeInterface
    v: NodeInterface
    action: TestResultAction
    data: Optional[Dict] = None


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
    def add_edge(self, u, v, metadata):
        pass

    @abstractmethod
    def remove_edge(self, u, v):
        pass

    @abstractmethod
    def remove_directed_edge(self, u, v):
        pass

    @abstractmethod
    def update_edge(self, u, v, metadata=None, edge_type=None):
        pass

    @abstractmethod
    def update_directed_edge(self, u, v, metadata=None, edge_type=None):
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
    def execute_pipeline_step(self, step, apply_to_graph: bool = True):
        pass


class GeneratorInterface(ABC, BaseModel):
    comparison_settings: Optional[ComparisonSettings] = None
    chunked: Optional[bool] = False
    every_nth: Optional[int] = None
    generator: Optional["GeneratorInterface"] = None
    shuffle_combinations: Optional[bool] = None

    @abstractmethod
    def generate(self, graph: BaseGraphInterface, graph_model_instance_: dict):
        pass

    def __init__(
        self,
        comparison_settings: Optional[ComparisonSettings] = None,
        chunked: bool = None,
        every_nth: int = None,
        generator: "GeneratorInterface" = None,
        shuffle_combinations: bool = None,
    ):
        super().__init__(comparison_settings=comparison_settings)
        if isinstance(comparison_settings, dict):
            comparison_settings = load_pipeline_artefact_by_definition(
                comparison_settings
            )

        if chunked is not None:
            self.chunked = chunked

        if every_nth is not None:
            self.every_nth = every_nth

        if generator is not None:
            self.generator = generator

        if shuffle_combinations is not None:
            self.shuffle_combinations = shuffle_combinations

        self.comparison_settings = comparison_settings

    @computed_field
    @property
    def name(self) -> str:
        return serialize_module_name(self)


PipelineStepInterfaceType = TypeVar("PipelineStepInterfaceType")


class PipelineStepInterface(ABC, BaseModel, Generic[PipelineStepInterfaceType]):
    generator: Optional[GeneratorInterface] = None
    threshold: Optional[float] = DEFAULT_THRESHOLD
    chunk_size_parallel_processing: int = 1
    parallel: bool = True

    display_name: Optional[str] = None

    def __init__(
        self,
        threshold: float = None,
        generator: Optional[GeneratorInterface] = None,
        chunk_size_parallel_processing: int = None,
        parallel: bool = None,
        display_name: Optional[str] = None,
    ):
        super().__init__()
        if generator:
            if isinstance(generator, dict):
                self.generator = load_pipeline_artefact_by_definition(generator)
            else:
                self.generator = generator

        if chunk_size_parallel_processing:
            self.chunk_size_parallel_processing = chunk_size_parallel_processing

        if parallel:
            self.parallel = parallel

        if display_name:
            self.display_name = display_name

        if threshold:
            self.threshold = threshold

    @computed_field
    @property
    def name(self) -> str:
        return serialize_module_name(self)

    @abstractmethod
    def test(self, nodes: List[str], graph: BaseGraphInterface) -> Optional[TestResult]:
        """
        Test if u and v are independent
        :param u: u values
        :param v: v values
        :return: True if independent, False otherwise
        """
        pass

    def __call__(
        self, nodes: List[str], graph: BaseGraphInterface
    ) -> Optional[TestResult]:
        return self.test(nodes, graph)


class ExitConditionInterface(ABC, BaseModel):
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

    @computed_field
    @property
    def name(self) -> str:
        return serialize_module_name(self)


LogicStepInterfaceType = TypeVar("LogicStepInterfaceType")


class LogicStepInterface(ABC, BaseModel, Generic[LogicStepInterfaceType]):
    pipeline_steps: Optional[List[Union[PipelineStepInterfaceType]]] = None
    exit_condition: Optional[ExitConditionInterface] = None

    display_name: Optional[str] = None

    @abstractmethod
    def execute(self, graph: BaseGraphInterface, graph_model_instance_: dict):
        pass

    @computed_field
    @property
    def name(self) -> str:
        return serialize_module_name(self)

    def __init__(
        self,
        pipeline_steps: Optional[
            Union[List[PipelineStepInterfaceType], Dict[Any, Any]]
        ] = None,
        exit_condition: Union[ExitConditionInterface, Dict[Any, Any]] = None,
        display_name: Optional[str] = None,
    ):
        super().__init__()
        # TODO check if this is a good idea
        if isinstance(exit_condition, dict):
            exit_condition = load_pipeline_artefact_by_definition(exit_condition)

        # TODO: check if this is a good idea
        if len(pipeline_steps) > 0 and isinstance(pipeline_steps[0], dict):
            pipeline_steps = load_pipeline_steps_by_definition(pipeline_steps)

        self.pipeline_steps = pipeline_steps or []
        self.exit_condition = exit_condition

        if display_name:
            self.display_name = display_name


CausyExtensionType = TypeVar("CausyExtensionType", bound="CausyExtension")


class CausyExtension(BaseModel, Generic[CausyExtensionType]):
    @computed_field
    @property
    def name(self) -> str:
        return serialize_module_name(self)


class CausyAlgorithm(BaseModel):
    name: str
    pipeline_steps: List[Union[PipelineStepInterfaceType, LogicStepInterface]]
    pipeline_steps: List[Union[PipelineStepInterfaceType, LogicStepInterface]]
    edge_types: List[EdgeTypeInterface]
    extensions: Optional[List[CausyExtensionType]] = None


class CausyAlgorithmReferenceType(enum.StrEnum):
    FILE = "file"
    NAME = "name"
    PYTHON_MODULE = "python_module"


class CausyAlgorithmReference(BaseModel):
    reference: str
    type: CausyAlgorithmReferenceType


class ActionHistoryStep(BaseModel):
    name: str
    duration: Optional[float] = None  # seconds
    actions: Optional[List[TestResult]] = []
    steps: Optional[List["ActionHistoryStep"]] = []


class CausyResult(BaseModel):
    algorithm: CausyAlgorithmReference
    created_at: datetime = Field(default_factory=datetime.now)
    nodes: Dict[str, NodeInterface]
    edges: List[EdgeInterface]
    action_history: List[ActionHistoryStep]
