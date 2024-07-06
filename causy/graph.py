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


class GraphAccessMixin:
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

    def only_directed_edge_exists(
        self, u: Union[Node, str], v: Union[Node, str]
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

    def __resolve_node_references(
        self, u: Union[Node, str], v: Optional[Union[Node, str]] = None
    ) -> Union[Node, Tuple[Node, Node]]:
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

        if not self.edge_exists(u, v):
            return False

        if self.edges[u][v].edge_type != edge_type:
            return False

        return True

    def directed_paths(
        self, u: Union[Node, str], v: Union[Node, str]
    ) -> List[List[Tuple[Node, Node]]]:
        """
        Return all directed paths from u to v
        :param u: node u
        :param v: node v
        :return: list of directed paths
        """
        u, v = self.__resolve_node_references(u, v)
        # TODO: try a better data structure for this
        if self.directed_edge_exists(u, v):
            return [[(u, v)]]
        paths = []
        for w in self.edges[u.id]:
            if self.directed_edge_exists(u, self.nodes[w]):
                for path in self.directed_paths(self.nodes[w], v):
                    paths.append([(u, self.nodes[w])] + path)
        return paths

    def inducing_path_exists(self, u: Union[Node, str], v: Union[Node, str]) -> bool:
        """
        Check if an inducing path from u to v exists.
        An inducing path from u to v is a directed reference from u to v on which all mediators are colliders.
        :param u: node u
        :param v: node v
        :return: True if an inducing path exists, False otherwise
        """

        if isinstance(u, Node):
            u = u.id
        if isinstance(v, Node):
            v = v.id

        if not self.directed_path_exists(u, v):
            return False
        for path in self.directed_paths(u, v):
            for i in range(1, len(path) - 1):
                r, w = path[i]
                if not self.bidirected_edge_exists(r, w):
                    # TODO: check if this is correct (@sof)
                    return True
        return False

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


class Graph(BaseModel, GraphAccessMixin):
    nodes: OrderedDict[str, Node] = collections.OrderedDict({})
    edges: Dict[str, Dict[str, Edge]] = dict()
    _reverse_edges: Dict[str, Dict[str, Edge]] = dict()
    _deleted_edges: Dict[str, Dict[str, Edge]] = dict()
    edge_history: Dict[Tuple[str, str], List[TestResult]] = dict()
    action_history: List[ActionHistoryStep] = []


class GraphManager(GraphAccessMixin, BaseGraphInterface):
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

    def __init__(self):
        self.graph = Graph()
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

    def add_edge(self, u: Node, v: Node, metadata: Dict):
        """
        Add an edge to the graph
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

        a_edge = Edge(u=u, v=v, edge_type=UndirectedEdge(), metadata=metadata)
        self.edges[u.id][v.id] = a_edge
        self._reverse_edges[v.id][u.id] = a_edge

        b_edge = Edge(u=v, v=u, edge_type=UndirectedEdge(), metadata=metadata)
        self.edges[v.id][u.id] = b_edge
        self._reverse_edges[u.id][v.id] = b_edge

        self.edge_history[(u.id, v.id)] = []
        self.edge_history[(v.id, u.id)] = []

    def add_directed_edge(self, u: Node, v: Node, metadata: Dict):
        """
        Add a directed edge from u to v to the graph
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

        edge = Edge(u=u, v=v, edge_type=DirectedEdge(), metadata=metadata)

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
