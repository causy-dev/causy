import multiprocessing
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, TypeVar, Generic, Any
from typing_extensions import Annotated
import logging

from pydantic import BaseModel, computed_field, Field, PlainValidator, WithJsonSchema
import torch

from causy.graph_utils import (
    load_pipeline_artefact_by_definition,
    serialize_module_name,
    load_pipeline_steps_by_definition,
)
from causy.variables import (
    StringParameter,
    IntegerParameter,
    BoolParameter,
    FloatParameter,
)

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.01

AS_MANY_AS_FIELDS = 0

MetadataBaseType = Union[str, int, float, bool]
MetadataType = Union[
    str, int, float, bool, List[MetadataBaseType], Dict[str, MetadataBaseType]
]


class ComparisonSettingsInterface(BaseModel, ABC):
    min: IntegerParameter
    max: IntegerParameter

    @computed_field
    @property
    def name(self) -> str:
        return serialize_module_name(self)


class NodeInterface(BaseModel, ABC):
    """
    Node interface for the graph. A node is defined by a name and a value.
    """

    name: str
    id: str
    values: Annotated[
        Optional[torch.DoubleTensor],
        WithJsonSchema(
            {
                "type": "array",
                "items": {"type": "number"},
                "description": "Node values",
            }
        ),
    ] = Field(exclude=True, default=None)

    class Config:
        arbitrary_types_allowed = True


EdgeTypeInterfaceType = TypeVar("EdgeTypeInterfaceType")


class EdgeTypeInterface(ABC, BaseModel, Generic[EdgeTypeInterfaceType]):
    """
    Edge type interface for the graph
    An edge type is defined by a name
    """

    name: str

    # define if it is a directed or undirected edge type (default is undirected). We use this e.g. when we compare the graph.
    IS_DIRECTED: bool = True
    STR_REPRESENTATION: str = "-"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class EdgeInterface(BaseModel, ABC):
    """
    Edge interface for the graph
    A graph edge is defined by two nodes and an edge type. It can also have metadata.
    """

    u: NodeInterface
    v: NodeInterface
    edge_type: EdgeTypeInterface
    metadata: Dict[str, MetadataType] = None

    class Config:
        arbitrary_types_allowed = True

    def __eq__(self, other):
        """
        Check if two edges are equal by comparing the nodes and the edge type
        :param other:
        :return:
        """
        if not isinstance(other, EdgeInterface):
            return False

        if self.edge_type != other.edge_type:
            return False
        if self.edge_type.IS_DIRECTED:
            return self.u == other.u and self.v == other.v
        else:
            return self.is_connection_between_same_nodes(other)

    def is_connection_between_same_nodes(self, edge):
        return (
            self.u == edge.u
            and self.v == edge.v
            or self.u == edge.v
            and self.v == edge.u
        )


class TestResultInterface(BaseModel, ABC):
    """
    Test result interface for the graph
    A test result is defined by two nodes and an action. It can also have metadata.
    """

    u: NodeInterface
    v: NodeInterface
    action: str
    data: Optional[Dict] = None

    class Config:
        arbitrary_types_allowed = True


class BaseGraphInterface(ABC):
    nodes: Dict[str, NodeInterface]
    edges: Dict[str, Dict[str, Dict]]

    @abstractmethod
    def retrieve_edge_history(self, u, v, action: str) -> List[TestResultInterface]:
        pass

    @abstractmethod
    def add_edge_history(self, u, v, action: TestResultInterface):
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
    def edge_exists(self, u, v, ignore_soft_deleted=False):
        pass

    @abstractmethod
    def restore_edge(self, u, v):
        pass

    @abstractmethod
    def restore_directed_edge(self, u, v):
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
    comparison_settings: Optional[ComparisonSettingsInterface] = None
    chunked: Optional[BoolParameter] = False
    every_nth: Optional[IntegerParameter] = None
    generator: Optional["GeneratorInterface"] = None
    shuffle_combinations: Optional[BoolParameter] = None

    @abstractmethod
    def generate(self, graph: BaseGraphInterface, graph_model_instance_: dict):
        pass

    def __init__(
        self,
        comparison_settings: Optional[ComparisonSettingsInterface] = None,
        chunked: Optional[BoolParameter] = None,
        every_nth: Optional[IntegerParameter] = None,
        generator: Optional["GeneratorInterface"] = None,
        shuffle_combinations: Optional[BoolParameter] = None,
    ) -> None:
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
    threshold: Optional[FloatParameter] = DEFAULT_THRESHOLD
    chunk_size_parallel_processing: IntegerParameter = 1
    parallel: BoolParameter = True

    display_name: Optional[StringParameter] = None

    needs_unapplied_actions: Optional[BoolParameter] = False

    def __init__(
        self,
        threshold: Optional[FloatParameter] = None,
        generator: Optional[GeneratorInterface] = None,
        chunk_size_parallel_processing: Optional[IntegerParameter] = None,
        parallel: Optional[BoolParameter] = None,
        display_name: Optional[StringParameter] = None,
        **kwargs,
    ) -> None:
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
    def process(
        self,
        nodes: List[str],
        graph: BaseGraphInterface,
        unapplied_actions: Optional[List[TestResultInterface]] = None,
    ) -> Optional[TestResultInterface]:
        """
        Test if u and v are independent
        :param u: u values
        :param v: v values
        :return: True if independent, False otherwise
        """
        pass

    def __call__(
        self,
        nodes: List[str],
        graph: BaseGraphInterface,
        unapplied_actions: Optional[List[TestResultInterface]] = None,
    ) -> Optional[TestResultInterface]:
        if self.needs_unapplied_actions and unapplied_actions is None:
            logger.warn(
                f"Pipeline step {self.name} needs unapplied actions but none were provided"
            )
        elif self.needs_unapplied_actions and unapplied_actions is not None:
            return self.process(nodes, graph, unapplied_actions)

        return self.process(nodes, graph)


class ExitConditionInterface(ABC, BaseModel):
    @abstractmethod
    def check(
        self,
        graph: BaseGraphInterface,
        graph_model_instance_: GraphModelInterface,
        actions_taken: List[TestResultInterface],
        iteration: IntegerParameter,
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
        actions_taken: List[TestResultInterface],
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

    display_name: Optional[StringParameter] = None

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


ExtensionType = TypeVar("ExtensionType", bound="ExtensionInterface")


class ExtensionInterface(BaseModel, Generic[ExtensionType]):
    @computed_field
    @property
    def name(self) -> str:
        return serialize_module_name(self)
