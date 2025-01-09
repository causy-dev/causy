import collections
from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple, Union, OrderedDict, Any
from uuid import uuid4
import logging

import torch
from pydantic import BaseModel, Field

from causy.edge_types import UndirectedEdge, DirectedEdge
from causy.interfaces import (
    BaseGraphInterface,
    NodeInterface,
    EdgeInterface,
    EdgeTypeInterface,
    MetadataType,
)
from causy.models import TestResultAction, TestResult, ActionHistoryStep
from causy.variables import VariableType

logger = logging.getLogger(__name__)


class Node(NodeInterface):
    """
    A node is a variable in the graph. It has a name, an id and values. The values are stored as a torch.Tensor.
    A node can also have metadata.
    """

    name: str
    id: str
    values: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, MetadataType]] = None

    def __hash__(self):
        return hash(self.id)


@dataclass
class Edge(EdgeInterface):
    """
    An edge is a connection between two nodes. It has a direction defined by the edge_type and metadata.
    A metadata example could be the p-value of a statistical test.
    An edge can be e.g. undirected, directed or bidirected. Which option is available/chosen depends on the algorithm and the current state of the graph.
    By default an edge always has an undirected and a directed edge_type.
    """

    u: NodeInterface
    v: NodeInterface
    edge_type: EdgeTypeInterface
    metadata: Optional[Dict[str, MetadataType]] = None
    deleted: Optional[bool] = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __hash__(self):
        return hash((self.u.id, self.v.id))


class GraphError(Exception):
    pass


class GraphBaseAccessMixin:
    def directed_edge_is_soft_deleted(
        self, u: Union[Node, str], v: Union[Node, str]
    ) -> bool:
        """
        Check if an edge is soft deleted
        :param u:
        :param v:
        :return:
        """

        if isinstance(u, Node):
            u = u.id
        if isinstance(v, Node):
            v = v.id

        if u not in self.edges:
            return False
        if v not in self.edges[u]:
            return False
        return self.edges[u][v].deleted

    def parents_of_node(self, u: Union[Node, str]):
        """
        Return all parents of a node u
        :param u: node u
        :return: list of nodes (parents)
        """
        if isinstance(u, Node):
            u = u.id

        return [
            self.nodes[n]
            for n in self._reverse_edges[u].keys()
            if not self.directed_edge_is_soft_deleted(n, u)
        ]

    def edge_exists(self, u: Union[Node, str], v: Union[Node, str]) -> bool:
        """
        Check if any edge exists between u and v. Cases: u -> v, u <-> v, u <- v
        :param u: node u
        :param v: node v
        :return: True if any edge exists, False otherwise
        """

        if isinstance(u, Node):
            u = u.id
        if isinstance(v, Node):
            v = v.id

        if u not in self.nodes:
            return False
        if v not in self.nodes:
            return False
        if (
            u in self.edges
            and v in self.edges[u]
            and not self.directed_edge_is_soft_deleted(u, v)
        ):
            return True
        if (
            v in self.edges
            and u in self.edges[v]
            and not self.directed_edge_is_soft_deleted(v, u)
        ):
            return True
        return False

    def directed_edge_exists(self, u: Union[Node, str], v: Union[Node, str]) -> bool:
        """
        Check if a directed edge exists between u and v. Cases: u -> v, u <-> v
        :param u: node u
        :param v: node v
        :return: True if a directed edge exists, False otherwise
        """

        if isinstance(u, Node):
            u = u.id
        if isinstance(v, Node):
            v = v.id

        if u not in self.nodes:
            return False
        if v not in self.nodes:
            return False
        if u not in self.edges:
            return False
        if v not in self.edges[u]:
            return False

        if self.directed_edge_is_soft_deleted(u, v):
            return False

        return True

    def node_by_id(self, id_: str) -> Optional[Node]:
        """
        Retrieve a node by its id
        :param id_:
        :return:
        """
        return self.nodes.get(id_)

    def edge_value(self, u: Union[Node, str], v: Union[Node, str]) -> Optional[Dict]:
        """
        retrieve the value of an edge
        :param u:
        :param v:
        :return:
        """

        if isinstance(u, Node):
            u = u.id
        if isinstance(v, Node):
            v = v.id

        if not self.edge_exists(u, v):
            return None
        return self.edges[u][v].metadata

    def edge_type(
        self, u: Union[Node, str], v: Union[Node, str]
    ) -> Optional[EdgeTypeInterface]:
        """
        retrieve the value of an edge
        :param u:
        :param v:
        :return:
        """
        if isinstance(u, Node):
            u = u.id
        if isinstance(v, Node):
            v = v.id
        if not self.edge_exists(u, v):
            return None
        return self.edges[u][v].edge_type

    def undirected_edge_exists(self, u: Union[Node, str], v: Union[Node, str]) -> bool:
        """
        Check if an undirected edge exists between u and v. Note: currently, an undirected edges is implemented just as
        a directed edge. However, they are two functions as they mean different things in different algorithms.
        Currently, this function is used in the PC algorithm, where an undirected edge is an edge which could not be
        oriented in any direction by orientation rules.
        Later, a cohersive naming scheme should be implemented.
        :param u: node u
        :param v: node v
        :return: True if an undirected edge exists, False otherwise
        """

        if self.directed_edge_exists(u, v) and self.directed_edge_exists(v, u):
            return True
        return False

    def get_siblings(self, v: Union[Node, str]) -> Set[Union[Node, str]]:
        """
        Get the set of nodes that are connected to the node v with an undirected edge.
        :param v: node v
        :return: A set of nodes that are connected to v with an undirected edge.
        """
        siblings = set()

        # Assuming self.nodes is a list or set of all nodes in the graph
        for node in self.nodes:
            if node != v and self.undirected_edge_exists(v, node):
                siblings.add(node)

        return siblings

    def retrieve_edge_history(
        self, u: Union[Node, str], v: Union[Node, str], action: TestResultAction = None
    ) -> List[TestResult]:
        """
        Retrieve the edge history
        :param u:
        :param v:
        :param action:
        :return:
        """

        if isinstance(u, Node):
            u = u.id
        if isinstance(v, Node):
            v = v.id

        if action is None:
            return self.edge_history[(u, v)]

        if (u, v) not in self.edge_history:
            return []

        return [i for i in self.edge_history[(u, v)] if i.action == action]

    def directed_path_exists(self, u: Union[Node, str], v: Union[Node, str]) -> bool:
        """
        Check if a directed path from u to v exists
        :param u: node u
        :param v: node v
        :return: True if a directed path exists, False otherwise
        """

        if isinstance(u, Node):
            u = u.id
        if isinstance(v, Node):
            v = v.id

        if self.directed_edge_exists(u, v):
            return True
        for w in self.edges[u]:
            if self.directed_path_exists(self.nodes[w], v):
                return True
        return False

    def edge_of_type_exists(
        self,
        u: Union[Node, str],
        v: Union[Node, str],
        edge_type: EdgeTypeInterface = DirectedEdge(),
    ) -> bool:
        """
        Check if an edge of a specific type exists between u and v.
        :param u: node u
        :param v: node v
        :param edge_type: the type of the edge to check for
        :return: True if an edge of this type exists, False otherwise
        """

        if isinstance(u, Node):
            u = u.id
        if isinstance(v, Node):
            v = v.id

        if not self.directed_edge_exists(u, v):
            return False

        if self.edges[u][v].edge_type != edge_type:
            return False

        return True

    def descendants_of_node(
        self, u: Union[Node, str], visited=None
    ) -> List[Union[Node, str]]:
        """
        Returns a list of all descendants of the given node in a directed graph, including the input node itself.
        :param u: The node or node ID to find descendants for.
        :return: A list of descendants of a node including the input node itself.
        """

        if visited is None:
            visited = set()

        if isinstance(u, Node):
            u = u.id

        # If this node has already been visited, return an empty list to avoid cycles
        if u in visited:
            return []

        visited.add(u)
        descendants = [self.node_by_id(u)]

        if u not in self.edges:
            return descendants

        for child in self.edges[u]:
            if self.directed_edge_exists(u, child) and not self.directed_edge_exists(
                child, u
            ):
                if child not in visited:
                    descendants.extend(self.descendants_of_node(child, visited))

        return list(set(descendants))

    def _is_a_collider_blocking(self, path, conditioning_set) -> bool:
        """
        Check if a path is blocked by a collider which is not in the conditioning set and has no descendants in the conditioning set.
        :return: Boolean indicating if the path is blocked
        """
        is_path_blocked = False
        for i in range(1, len(path) - 1):
            if self.edge_of_type_exists(
                path[i - 1], path[i], DirectedEdge()
            ) and self.edge_of_type_exists(path[i + 1], path[i], DirectedEdge()):
                # if the node is a collider, check if the node or any of its descendants are in the conditioning set
                is_path_blocked = True
                for descendant in self.descendants_of_node(path[i]):
                    if descendant in conditioning_set:
                        is_path_blocked = False
        return is_path_blocked

    def _is_a_non_collider_in_conditioning_set(self, path, conditioning_set) -> bool:
        """
        Check if a path is blocked by a non-collider which is in the conditioning set.
        :param path:
        :param conditioning_set:
        :return:
        """
        is_path_blocked = False
        for i in range(1, len(path) - 1):
            if path[i] in conditioning_set:
                # make sure that node is a noncollider
                if not (
                    self.edge_of_type_exists(path[i - 1].id, path[i].id, DirectedEdge())
                    and self.edge_of_type_exists(
                        path[i + 1].id, path[i].id, DirectedEdge()
                    )
                ):
                    # if the node is a non-collider and in the conditioning set, the path is blocked
                    is_path_blocked = True
        return is_path_blocked

    def are_nodes_d_separated(
        self,
        u: Union[Node, str],
        v: Union[Node, str],
        conditioning_set: List[Union[Node, str]],
    ) -> bool:
        """
        Check if nodes u and v are d-separated given a conditioning set. We check whether there is an open path, i.e. a path on which all colliders are in the conditioning set and all non-colliders are not in the conditioning set. If there is no open path, u and v are d-separated.

        :param u: First node
        :param v: Second node
        :param conditioning_set: Set of nodes to condition on
        :return: True if u and v are d-separated given conditioning_set, False otherwise
        """

        # Convert Node instances to their IDs
        if isinstance(u, Node):
            u = u.id
        if isinstance(v, Node):
            v = v.id

        # u and v may not be in the conditioning set, throw error
        if u in conditioning_set or v in conditioning_set:
            raise ValueError("Nodes u and v may not be in the conditioning set")

        # If there are no paths between u and v, they are d-separated
        if list(self.all_paths_on_underlying_undirected_graph(u, v)) == []:
            return True

        list_of_results_for_paths = []
        for path in self.all_paths_on_underlying_undirected_graph(u, v):
            # If the path only has two nodes, it cannot be blocked and is open. Therefore, u and v are not d-separated
            if len(path) == 2:
                return False

            is_path_blocked = False

            if self._is_a_collider_blocking(
                path, conditioning_set
            ) or self._is_a_non_collider_in_conditioning_set(path, conditioning_set):
                is_path_blocked = True

            list_of_results_for_paths.append(is_path_blocked)

        # if there is at least one open path, u and v are not d-separated
        if False in list_of_results_for_paths:
            return False
        return True

    def all_paths_on_underlying_undirected_graph(
        self, u: Union[Node, str], v: Union[Node, str], visited=None, path=None
    ) -> List[List[Node]]:
        if isinstance(u, Node):
            u = u.id
        if isinstance(v, Node):
            v = v.id

        # Initialize visited and path lists during the first call
        if visited is None:
            visited = set()
        if path is None:
            path = []

        # Add the current node to the path and mark it as visited
        visited.add(u)
        path.append(u)

        # If the current node is the target node, yield the current path
        if u == v:
            yield list([self.node_by_id(node_id) for node_id in path])
        else:
            # Explore all possible next nodes by checking edge existence
            for next_node in self.nodes:
                if next_node not in visited and self.edge_exists(u, next_node):
                    yield from self.all_paths_on_underlying_undirected_graph(
                        next_node, v, visited, path
                    )

        # Backtrack: remove the current node from the path and visited set
        path.pop()
        visited.remove(u)

    def retrieve_edges(self) -> List[Edge]:
        """
        Retrieve all edges
        :return: all edges
        """
        edges = []
        for u in self.edges:
            for v in self.edges[u]:
                if not self.directed_edge_is_soft_deleted(self.nodes[u], self.nodes[v]):
                    edges.append(self.edges[u][v])
        return edges


class Graph(BaseModel, GraphBaseAccessMixin):
    nodes: OrderedDict[str, Node] = collections.OrderedDict({})
    edges: Dict[str, Dict[str, Edge]] = dict()
    _reverse_edges: Dict[str, Dict[str, Edge]] = dict()
    _deleted_edges: Dict[str, Dict[str, Edge]] = dict()
    edge_history: Dict[Tuple[str, str], List[TestResult]] = dict()
    action_history: List[ActionHistoryStep] = []


class GraphManager(GraphBaseAccessMixin, BaseGraphInterface):
    """
    The graph represents the internal data structure of causy. It is a simple graph with nodes and edges.
    But it supports to be handled as a directed graph, undirected graph and bidirected graph, which is important to implement different algorithms in different stages.
    It also stores the history of the actions taken on the graph.
    """

    @property
    def nodes(self) -> OrderedDict[str, Node]:
        return self.graph.nodes

    @property
    def edges(self) -> Dict[str, Dict[str, Edge]]:
        return self.graph.edges

    @property
    def _reverse_edges(self) -> Dict[str, Dict[str, Edge]]:
        return self.graph._reverse_edges

    @property
    def _deleted_edges(self) -> Dict[str, Dict[str, Edge]]:
        return self.graph._deleted_edges

    @property
    def edge_history(self) -> Dict[Tuple[str, str], List[TestResult]]:
        return self.graph.edge_history

    @property
    def action_history(self) -> List[Dict[str, List[TestResult]]]:
        return self.graph.action_history

    graph: Optional[Graph] = None

    def __init__(self, graph_class=Graph):
        self.graph = graph_class()
        self.graph.nodes = collections.OrderedDict({})
        self.graph.edges = self.__init_dict()
        self.graph._reverse_edges = self.__init_dict()
        self.graph.edge_history = self.__init_dict()
        self.graph.action_history = []

    def __init_dict(self):
        """
        Initialize a dictionary - we encapsulate this to make it easier to change the implementation of the dictionary
        As this might be necessary for multiprocessing
        :return:
        """
        return dict()

    def get_edge(self, u: Node, v: Node) -> Edge:
        """
        Get an edge between two nodes
        :param u: u node
        :param v: v node
        :return: the edge
        """
        if u.id not in self.edges:
            raise GraphError(f"Node {u} does not exist")
        if v.id not in self.edges[u.id]:
            raise GraphError(f"Edge {u} -> {v} does not exist")
        return self.edges[u.id][v.id]

    def add_edge(
        self,
        u: Node,
        v: Node,
        metadata: Dict,
        edge_type: EdgeTypeInterface = UndirectedEdge(),
    ):
        """
        Add an edge to the graph
        :param edge_type: the type of the edge (e.g. undirected, directed, bidirected)
        :param u: u node
        :param v: v node
        :param metadata: metadata of the edge
        :return:
        """

        if u.id not in self.nodes:
            raise GraphError(f"Node {u} does not exist")
        if v.id not in self.nodes:
            raise GraphError(f"Node {v} does not exist")

        if u.id == v.id:
            raise GraphError("Self loops are currently not allowed")

        if u.id not in self.edges:
            self.edges[u.id] = self.__init_dict()
            self._reverse_edges[u.id] = self.__init_dict()
            self._deleted_edges[u.id] = self.__init_dict()
        if v.id not in self.edges:
            self.edges[v.id] = self.__init_dict()
            self._reverse_edges[v.id] = self.__init_dict()
            self._deleted_edges[v.id] = self.__init_dict()

        a_edge = Edge(u=u, v=v, edge_type=edge_type, metadata=metadata)
        self.edges[u.id][v.id] = a_edge
        self._reverse_edges[v.id][u.id] = a_edge

        b_edge = Edge(u=v, v=u, edge_type=edge_type, metadata=metadata)
        self.edges[v.id][u.id] = b_edge
        self._reverse_edges[u.id][v.id] = b_edge

        self.edge_history[(u.id, v.id)] = []
        self.edge_history[(v.id, u.id)] = []

    def add_directed_edge(
        self,
        u: Node,
        v: Node,
        metadata: Dict,
        edge_type: EdgeTypeInterface = DirectedEdge(),
    ):
        """
        Add a directed edge from u to v to the graph
        :param edge_type:
        :param u: u node
        :param v: v node
        :param metadata: metadata of the edge
        :return:
        """

        if u.id not in self.nodes:
            raise GraphError(f"Node {u} does not exist")
        if v.id not in self.nodes:
            raise GraphError(f"Node {v} does not exist")

        if u.id == v.id:
            raise GraphError("Self loops are currently not allowed")

        if u.id not in self.edges:
            self.edges[u.id] = self.__init_dict()
            self._deleted_edges[u.id] = self.__init_dict()
        if v.id not in self._reverse_edges:
            self._reverse_edges[v.id] = self.__init_dict()

        edge = Edge(u=u, v=v, edge_type=edge_type, metadata=metadata)

        self.edges[u.id][v.id] = edge
        self._reverse_edges[v.id][u.id] = edge

        self.edge_history[(u.id, v.id)] = []

    def add_edge_history(self, u, v, action: TestResult):
        """
        Add an action to the edge history
        :param u:
        :param v:
        :param action:
        :return:
        """
        if (u.id, v.id) not in self.edge_history:
            self.edge_history[(u.id, v.id)] = []
        self.edge_history[(u.id, v.id)].append(action)

    def remove_edge(self, u: Node, v: Node, soft_delete: bool = False):
        """
        Remove an edge from the graph (undirected)
        :param u: u node
        :param v: v node
        :return:
        """
        if u.id not in self.nodes:
            raise GraphError(f"Node {u} does not exist")
        if v.id not in self.nodes:
            raise GraphError(f"Node {v} does not exist")

        if soft_delete:
            if u.id in self.edges and v.id in self.edges[u.id]:
                self.edges[u.id][v.id].deleted = True
                self._reverse_edges[v.id][u.id].deleted = True

            if v.id in self.edges and u.id in self.edges[v.id]:
                self.edges[v.id][u.id].deleted = True
                self._reverse_edges[u.id][v.id].deleted = True

            return

        if u.id in self.edges and v.id in self.edges[u.id]:
            self._deleted_edges[u.id][v.id] = self.edges[u.id][v.id]
            del self.edges[u.id][v.id]
            del self._reverse_edges[u.id][v.id]

        if v.id in self.edges and u.id in self.edges[v.id]:
            self._deleted_edges[v.id][u.id] = self.edges[v.id][u.id]
            del self.edges[v.id][u.id]
            del self._reverse_edges[v.id][u.id]

    def restore_edge(self, u: Node, v: Node):
        """
        Restore a deleted edge
        :param u:
        :param v:
        :return:
        """
        if u.id not in self.nodes:
            raise GraphError(f"Node {u} does not exist")
        if v.id not in self.nodes:
            raise GraphError(f"Node {v} does not exist")

        if u.id in self.edges and v.id in self.edges[u.id]:
            self.edges[u.id][v.id].deleted = False
            self._reverse_edges[v.id][u.id].deleted = False
        else:
            self.add_edge(u, v, self._deleted_edges[u.id][v.id].metadata)
            del self._deleted_edges[u.id][v.id]

        if v.id in self.edges and u.id in self.edges[v.id]:
            self.edges[v.id][u.id].deleted = False
            self._reverse_edges[u.id][v.id].deleted = False
        else:
            self.add_edge(v, u, self._deleted_edges[v.id][u.id].metadata)
            del self._deleted_edges[v.id][u.id]

    def remove_directed_edge(self, u: Node, v: Node, soft_delete: bool = False):
        """
        :param u:
        :param v:
        :param soft_delete: does not remove the edge, but marks it as deleted (useful in multithreading)
        :return:
        """
        if u.id not in self.nodes:
            raise GraphError(f"Node {u} does not exist")
        if v.id not in self.nodes:
            raise GraphError(f"Node {v} does not exist")

        if u.id not in self.edges:
            return  # no edges from u
        if v.id not in self.edges[u.id]:
            return

        if soft_delete:
            self.edges[u.id][v.id].deleted = True
            self._reverse_edges[v.id][u.id].deleted = True
            return

        self._deleted_edges[u.id][v.id] = self.edges[u.id][v.id]

        del self.edges[u.id][v.id]
        del self._reverse_edges[v.id][u.id]

    def restore_directed_edge(self, u: Node, v: Node):
        """
        Restore a soft deleted edge
        :param u:
        :param v:
        :return:
        """
        if u.id not in self.nodes:
            raise GraphError(f"Node {u} does not exist")
        if v.id not in self.nodes:
            raise GraphError(f"Node {v} does not exist")

        if u.id in self.edges and v.id in self.edges[u.id]:
            self.edges[u.id][v.id].deleted = False
            self._reverse_edges[v.id][u.id].deleted = False
        else:
            self.add_directed_edge(u, v, self._deleted_edges[u.id][v.id].metadata)
            del self._deleted_edges[u.id][v.id]

    def update_edge(
        self,
        u: Node,
        v: Node,
        metadata: Dict = None,
        edge_type: EdgeTypeInterface = None,
    ):
        """
        Update an undirected edge in the graph
        :param u: u node
        :param v: v node
        :param value:
        :param edge_type:
        :return:
        """

        if u.id not in self.nodes:
            raise GraphError(f"Node {u} does not exist")
        if v.id not in self.nodes:
            raise GraphError(f"Node {v} does not exist")
        if u.id not in self.edges:
            raise GraphError(f"Node {u} does not have any edges")
        if v.id not in self.edges:
            raise GraphError(f"Node {v} does not have any edges")

        if u.id not in self.edges[v.id]:
            raise GraphError(f"There is no edge from {u} to {v}")

        if v.id not in self.edges[u.id]:
            raise GraphError(f"There is no edge from {v} to {u}")

        if metadata is not None:
            obj = self.edges[u.id][v.id]
            obj.metadata = metadata
            self.edges[u.id][v.id] = obj

            obj = self.edges[v.id][u.id]
            obj.metadata = metadata
            self.edges[v.id][u.id] = obj

            obj = self._reverse_edges[u.id][v.id]
            obj.metadata = metadata
            self._reverse_edges[u.id][v.id] = obj

            obj = self._reverse_edges[v.id][u.id]
            obj.metadata = metadata
            self._reverse_edges[v.id][u.id] = obj

        if edge_type is not None:
            obj = self.edges[u.id][v.id]
            obj.edge_type = edge_type
            self.edges[u.id][v.id] = obj

            obj = self.edges[v.id][u.id]
            obj.edge_type = edge_type
            self.edges[v.id][u.id] = obj

            obj = self._reverse_edges[u.id][v.id]
            obj.edge_type = edge_type
            self._reverse_edges[u.id][v.id] = obj

            obj = self._reverse_edges[v.id][u.id]
            obj.edge_type = edge_type
            self._reverse_edges[v.id][u.id] = obj

    def update_directed_edge(
        self,
        u: Node,
        v: Node,
        metadata: Dict = None,
        edge_type: EdgeTypeInterface = None,
    ):
        """
        Update an edge in the graph
        :param u: u node
        :param v: v node
        :return:
        """
        if u.id not in self.nodes:
            raise GraphError(f"Node {u} does not exist")
        if v.id not in self.nodes:
            raise GraphError(f"Node {v} does not exist")
        if u.id not in self.edges:
            raise GraphError(f"Node {u} does not have any edges")
        if v.id not in self.edges[u.id]:
            raise GraphError(f"There is no edge from {u} to {v}")

        if metadata is not None:
            obj = self.edges[u.id][v.id]
            obj.metadata = metadata
            self.edges[u.id][v.id] = obj

            obj = self._reverse_edges[v.id][u.id]
            obj.metadata = metadata
            self._reverse_edges[v.id][u.id] = obj

        if edge_type is not None:
            obj = self.edges[u.id][v.id]
            obj.edge_type = edge_type
            self.edges[u.id][v.id] = obj

            obj = self._reverse_edges[v.id][u.id]
            obj.edge_type = edge_type
            self._reverse_edges[v.id][u.id] = obj

    def purge_soft_deleted_edges(self):
        """
        Remove all edges that are marked as soft deleted from the graph
        """
        edges_to_remove = []
        for u_id, edges in self.edges.items():
            for v_id, edge in edges.items():
                if edge.deleted is True:
                    edges_to_remove.append((self.nodes[u_id], self.nodes[v_id]))

        for u, v in edges_to_remove:
            self.remove_directed_edge(u, v)

    def add_node(
        self,
        name: str,
        values: Union[List[float], torch.Tensor],
        id_: str = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Node:
        """
        Add a node to the graph
        :param metadata: add metadata to the node
        :param name: name of the node
        :param values: values of the node
        :param id_: id_ of the node
        :param : node

        :return: created Node
        """
        if id_ is None:
            id_ = str(uuid4())

        if id_ in self.nodes:
            raise ValueError(f"Node with id {id_} already exists")

        if isinstance(values, torch.Tensor):
            tensor_values = values
        else:
            try:
                tensor_values = torch.tensor(values, dtype=torch.float64)
            except TypeError as e:
                raise ValueError(f"Currently only numeric values are supported. {e}")

        if metadata is None:
            metadata = {}

        node = Node(name=name, id=id_, values=tensor_values, metadata=metadata)

        self.nodes[id_] = node
        return node
