import enum
from typing import Optional, List, Generic

from causy.contrib.graph_ui import (
    EdgeTypeConfig,
    EdgeUIConfig,
    ConditionalEdgeUIConfig,
    ConditionalEdgeUIConfigComparison,
)
from causy.interfaces import (
    EdgeTypeInterface,
    EdgeTypeInterfaceType,
)


class EdgeTypeEnum(enum.StrEnum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    BIDIRECTED = "bidirected"


class DirectedEdge(EdgeTypeInterface, Generic[EdgeTypeInterfaceType]):
    name: str = EdgeTypeEnum.DIRECTED.name
    IS_DIRECTED: bool = True
    STR_REPRESENTATION: str = "-->"  # u --> v


class DirectedEdgeUIConfig(EdgeTypeConfig):
    edge_type: str = DirectedEdge().name

    default_ui_config: Optional[EdgeUIConfig] = EdgeUIConfig(
        label_field="correlation",
        color="#333",
        width=4,
        style="dashed",
        animated=True,
        marker_start=None,
        marker_end="ArrowClosed",
        label="Direct Effect: ${correlation.toFixed(4)}",
    )
    conditional_ui_configs: Optional[List[ConditionalEdgeUIConfig]] = [
        ConditionalEdgeUIConfig(
            ui_config=EdgeUIConfig(
                color="#f00000",
                width=4,
                style="dashed",
                animated=True,
                marker_start=None,
                marker_end="ArrowClosed",
                label_field="direct_effect",
                label="Direct Effect: ${direct_effect.toFixed(4)}",
            ),
            condition_field="direct_effect",
            condition_value=0,
            condition_comparison=ConditionalEdgeUIConfigComparison.LESS,
        ),
        ConditionalEdgeUIConfig(
            ui_config=EdgeUIConfig(
                color="#0f0fff",
                width=4,
                style="dashed",
                animated=True,
                marker_start=None,
                marker_end="ArrowClosed",
                label_field="direct_effect",
                label="Direct Effect: ${direct_effect.toFixed(4)}",
            ),
            condition_field="direct_effect",
            condition_value=0,
            condition_comparison=ConditionalEdgeUIConfigComparison.GREATER,
        ),
    ]


class UndirectedEdge(EdgeTypeInterface, Generic[EdgeTypeInterfaceType]):
    name: str = EdgeTypeEnum.UNDIRECTED.name
    IS_DIRECTED: bool = False
    STR_REPRESENTATION: str = "---"  # u --- v


class UndirectedEdgeUIConfig(EdgeTypeConfig):
    edge_type: str = UndirectedEdge().name

    default_ui_config: Optional[EdgeUIConfig] = EdgeUIConfig(
        label_field="direct_effect",
        color="#333",
        width=4,
        style="solid",
        animated=False,
        marker_start=None,
        marker_end=None,
        label="Correlation: ${correlation.toFixed(4)}",
    )


class BiDirectedEdge(EdgeTypeInterface, Generic[EdgeTypeInterfaceType]):
    name: str = EdgeTypeEnum.BIDIRECTED.name
    IS_DIRECTED: bool = False  # This is a bi-directed edge - so it is not directed in the traditional sense
    STR_REPRESENTATION: str = "<->"  # u <-> v


class BiDirectedEdgeUIConfig(EdgeTypeConfig):
    edge_type: str = BiDirectedEdge().name

    default_ui_config: Optional[EdgeUIConfig] = EdgeUIConfig(
        label_field="direct_effect",
        color="#333",
        width=4,
        style="solid",
        animated=False,
        marker_start="ArrowClosed",
        marker_end="ArrowClosed",
    )


EDGE_TYPES = {
    DirectedEdge().name: DirectedEdge,
    UndirectedEdge().name: UndirectedEdge,
    BiDirectedEdge().name: BiDirectedEdge,
}
