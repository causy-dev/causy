import enum
from typing import Optional, List, Generic

from pydantic import BaseModel

from causy.interfaces import ExtensionInterface, ExtensionType


class EdgeUIConfig(BaseModel):
    color: Optional[str] = None
    width: Optional[int] = None
    style: Optional[str] = None
    marker_start: Optional[str] = None
    marker_end: Optional[str] = None
    label_field: Optional[str] = None
    animated: Optional[bool] = False
    label: Optional[str] = None


class ConditionalEdgeUIConfigComparison(enum.StrEnum):
    EQUAL = "EQUAL"
    GREATER = "GREATER"
    LESS = "LESS"
    GREATER_EQUAL = "GREATER_EQUAL"
    LESS_EQUAL = "LESS_EQUAL"
    NOT_EQUAL = "NOT_EQUAL"


class ConditionalEdgeUIConfig(BaseModel):
    ui_config: Optional[EdgeUIConfig] = None
    condition_field: str = "coefficient"
    condition_value: float = 0.5
    condition_comparison: ConditionalEdgeUIConfigComparison = (
        ConditionalEdgeUIConfigComparison.GREATER_EQUAL
    )


class EdgeTypeConfig(BaseModel):
    edge_type: str
    default_ui_config: Optional[EdgeUIConfig] = None
    conditional_ui_configs: Optional[List[ConditionalEdgeUIConfig]] = None


class GraphUIExtension(ExtensionInterface[ExtensionType], Generic[ExtensionType]):
    edges: List[EdgeTypeConfig]
