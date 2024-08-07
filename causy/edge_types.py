import enum
from typing import Optional, List, Generic, Union, Tuple

from causy.contrib.graph_ui import (
    EdgeTypeConfig,
    EdgeUIConfig,
    ConditionalEdgeUIConfig,
    ConditionalEdgeUIConfigComparison,
)
from causy.interfaces import (
    EdgeTypeInterface,
    EdgeTypeInterfaceType,
    NodeInterface,
)


class EdgeTypeEnum(str, enum.Enum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    BIDIRECTED = "bidirected"


class DirectedEdge(EdgeTypeInterface, Generic[EdgeTypeInterfaceType]):
    name: str = EdgeTypeEnum.DIRECTED.name
    IS_DIRECTED: bool = True
    STR_REPRESENTATION: str = "-->"  # u --> v

    class GraphAccessMixin:
        def only_directed_edge_exists(
            self, u: Union[NodeInterface, str], v: Union[NodeInterface, str]
        ) -> bool:
            """
            Check if a directed edge exists between u and v, but no directed edge exists between v and u. Case: u -> v
            :param u: node u
            :param v: node v
            :return: True if only directed edge exists, False otherwise
            """
            if self.directed_edge_exists(u, v) and not self.directed_edge_exists(v, u):
                return True
            return False

        def _resolve_node_references(
            self,
            u: Union[NodeInterface, str],
            v: Optional[Union[NodeInterface, str]] = None,
        ) -> Union[NodeInterface, Tuple[NodeInterface, NodeInterface]]:
            """
            Resolve node references
            :param u:
            :param v:
            :return: Returns a tuple of nodes if v is not None, otherwise returns a single node
            """
            if isinstance(u, str):
                u = self.node_by_id(u)
            if v and isinstance(v, str):
                v = self.node_by_id(v)

            if v is None:
                return u

            return u, v

        def directed_paths(
            self, u: Union[NodeInterface, str], v: Union[NodeInterface, str]
        ) -> List[List[Tuple[NodeInterface, NodeInterface]]]:
            """
            Return all directed paths from u to v
            :param u: node u
            :param v: node v
            :return: list of directed paths
            """
            u, v = self._resolve_node_references(u, v)
            # TODO: try a better data structure for this
            if self.directed_edge_exists(u, v):
                return [[(u, v)]]
            paths = []
            for w in self.edges[u.id]:
                if self.directed_edge_exists(u, self.nodes[w]):
                    for path in self.directed_paths(self.nodes[w], v):
                        paths.append([(u, self.nodes[w])] + path)
            return paths


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
