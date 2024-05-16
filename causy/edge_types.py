from typing import Optional, List

from causy.contrib.graph_ui import (
    EdgeTypeConfig,
    EdgeUIConfig,
    ConditionalEdgeUIConfig,
    ConditionalEdgeUIConfigComparison,
)
from causy.interfaces import (
    EdgeTypeInterface,
)


class DirectedEdge(EdgeTypeInterface):
    name: str = "DIRECTED"


class DirectedEdgeUIConfig(EdgeTypeConfig):
    edge_type: str = DirectedEdge().name

    default_ui_config: Optional[EdgeUIConfig] = EdgeUIConfig(
        label_field="direct_effect",
        color="#0f0fff",
        width=4,
        style="dashed",
        animated=True,
        marker_start=None,
        marker_end="ArrowClosed",
        label="Direct Effect: ${direct_effect.toFixed(4)}",
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
        )
    ]


class UndirectedEdge(EdgeTypeInterface):
    name: str = "UNDIRECTED"


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


class BiDirectedEdge(EdgeTypeInterface):
    name: str = "BIDIRECTED"


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
