from typing import Optional, List

from causy.interfaces import (
    EdgeTypeInterface,
    EdgeUIConfig,
    ConditionalEdgeUIConfig,
    ConditionalEdgeUIConfigComparison,
)


class DirectedEdge(EdgeTypeInterface):
    name: str = "DIRECTED"
    default_ui_config: Optional[EdgeUIConfig] = EdgeUIConfig(
        label_field="direct_effect",
        color="#0f0ff",
        width=4,
        style="solid",
        animated=True,
        marker_start=None,
        marker_end="ArrowClosed",
    )
    conditional_ui_configs: Optional[List[ConditionalEdgeUIConfig]] = [
        ConditionalEdgeUIConfig(
            ui_config=EdgeUIConfig(
                color="#ff0000",
                width=4,
                style="dashed",
                animated=False,
                marker_start=None,
                marker_end="ArrowClosed",
                label_field="direct_effect",
            ),
            condition_field="direct_effect",
            condition_value=0,
            condition_comparison=ConditionalEdgeUIConfigComparison.LESS,
        )
    ]


class UndirectedEdge(EdgeTypeInterface):
    name: str = "UNDIRECTED"
    default_ui_config: Optional[EdgeUIConfig] = EdgeUIConfig(
        label_field="direct_effect",
        color="#333",
        width=4,
        style="solid",
        animated=False,
        marker_start=None,
        marker_end=None,
    )


class BiDirectedEdge(EdgeTypeInterface):
    name: str = "BIDIRECTED"
    default_ui_config: Optional[EdgeUIConfig] = EdgeUIConfig(
        label_field="direct_effect",
        color="#333",
        width=4,
        style="solid",
        animated=False,
        marker_start="ArrowClosed",
        marker_end="ArrowClosed",
    )
