import enum
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, List, Union, Any

from pydantic import BaseModel, computed_field, Field

from causy.graph_utils import serialize_module_name, hash_dictionary
from causy.interfaces import (
    AS_MANY_AS_FIELDS,
    NodeInterface,
    PipelineStepInterfaceType,
    LogicStepInterface,
    EdgeTypeInterface,
    ExtensionType,
    EdgeInterface,
    TestResultInterface,
    ComparisonSettingsInterface,
    ExtensionInterface,
    EdgeTypeInterfaceType,
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

    UPDATE_EDGE = "UPDATE_EDGE"
    UPDATE_EDGE_DIRECTED = "UPDATE_EDGE_DIRECTED"

    UPDATE_EDGE_TYPE = "UPDATE_EDGE_TYPE"
    UPDATE_EDGE_TYPE_DIRECTED = "UPDATE_EDGE_TYPE_DIRECTED"

    REMOVE_EDGE_UNDIRECTED = "REMOVE_EDGE_UNDIRECTED"
    REMOVE_EDGE_DIRECTED = "REMOVE_EDGE_DIRECTED"

    RESTORE_EDGE = "RESTORE_EDGE"
    RESTORE_EDGE_DIRECTED = "RESTORE_EDGE_DIRECTED"

    DO_NOTHING = "DO_NOTHING"


class TestResult(TestResultInterface):
    u: NodeInterface
    v: NodeInterface
    action: TestResultAction
    data: Optional[Dict] = None


class AlgorithmReferenceType(enum.StrEnum):
    FILE = "file"
    NAME = "name"
    PYTHON_MODULE = "python_module"


class AlgorithmReference(BaseModel):
    reference: str
    type: AlgorithmReferenceType


class Algorithm(BaseModel):
    name: str
    pipeline_steps: List[Union[PipelineStepInterfaceType, LogicStepInterface]]
    pipeline_steps: List[Union[PipelineStepInterfaceType, LogicStepInterface]]
    edge_types: List[EdgeTypeInterfaceType]
    extensions: Optional[List[ExtensionType]] = None
    variables: Optional[List[Union[VariableInterfaceType]]] = None

    def hash(self) -> str:
        return hash_dictionary(self.model_dump())


class ActionHistoryStep(BaseModel):
    name: str
    duration: Optional[float] = None  # seconds
    actions: Optional[List[TestResult]] = []
    steps: Optional[List["ActionHistoryStep"]] = []


class Result(BaseModel):
    algorithm: AlgorithmReference
    created_at: datetime = Field(default_factory=datetime.now)
    nodes: Dict[str, NodeInterface]
    edges: List[EdgeInterface]
    action_history: List[ActionHistoryStep]
    variables: Optional[Dict[str, Any]] = None
    data_loader_hash: Optional[str] = None
    algorithm_hash: Optional[str] = None
    variables_hash: Optional[str] = None
