from typing import Optional, List, Dict, Union

from causy.interfaces import NodeInterface
from causy.models import Result
from causy.workspaces.models import Experiment
from pydantic import BaseModel, UUID4


class NodePosition(BaseModel):
    x: Optional[float]
    y: Optional[float]


class ExperimentVersion(BaseModel):
    version: int
    name: str


class ExtendedExperiment(Experiment):
    versions: Optional[List[ExperimentVersion]] = None
    name: str = None


class PositionedNode(NodeInterface):
    position: Optional[NodePosition] = None


class ExtendedResult(Result):
    nodes: Dict[Union[UUID4, str], PositionedNode]
    version: Optional[int] = None
