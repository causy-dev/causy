import enum
from datetime import datetime
from typing import Optional, Dict, List, Union

from pydantic import BaseModel, computed_field, Field

from causy.graph_utils import serialize_module_name
from causy.interfaces import (
    AS_MANY_AS_FIELDS,
    NodeInterface,
    PipelineStepInterfaceType,
    LogicStepInterface,
    EdgeTypeInterface,
    CausyExtensionType,
    EdgeInterface,
    TestResultInterface,
    ComparisonSettingsInterface,
    CausyExtensionInterface,
)
from causy.variables import IntegerParameter, VariableInterfaceType


class ComparisonSettings(ComparisonSettingsInterface):
    min: IntegerParameter = 2
    max: IntegerParameter = AS_MANY_AS_FIELDS

    @computed_field
    @property
    def name(self) -> str:
        return serialize_module_name(self)


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


class TestResult(TestResultInterface):
    u: NodeInterface
    v: NodeInterface
    action: TestResultAction
    data: Optional[Dict] = None


class CausyAlgorithmReferenceType(enum.StrEnum):
    FILE = "file"
    NAME = "name"
    PYTHON_MODULE = "python_module"


class CausyAlgorithmReference(BaseModel):
    reference: str
    type: CausyAlgorithmReferenceType


class CausyAlgorithm(BaseModel):
    name: str
    pipeline_steps: List[Union[PipelineStepInterfaceType, LogicStepInterface]]
    pipeline_steps: List[Union[PipelineStepInterfaceType, LogicStepInterface]]
    edge_types: List[EdgeTypeInterface]
    extensions: Optional[List[CausyExtensionInterface]] = None
    variables: Optional[List[Union[VariableInterfaceType]]] = None


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
