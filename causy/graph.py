from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple, Union
from uuid import uuid4
import logging

import torch

from causy.interfaces import (
    BaseGraphInterface,
    NodeInterface,
    TestResultAction,
    TestResult,
)

logger = logging.getLogger(__name__)


@dataclass
class Node(NodeInterface):
    name: str
    id: str
    values: torch.Tensor
    metadata: Dict[str, any] = None

    def __hash__(self):
        return hash(self.id)


class GraphError(Exception):
    pass


class Graph(BaseGraphInterface):
    """
    The graph represents the internal data structure of causy. It is a simple graph with nodes and edges.
    But it supports to be handled as a directed graph, undirected graph and bidirected graph, which is important to implement different algorithms in different stages.
    It also stores the history of the actions taken on the graph.
    """

    nodes: Dict[str, Node]
    edges: Dict[str, Dict[str, Dict]]
    _reverse_edges: Dict[str, Dict[str, Dict]]
    edge_history: Dict[Tuple[str, str], List[TestResult]]
    action_history: List[Dict[str, List[TestResult]]]

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self._reverse_edges = {}
        self.edge_history = {}
        self.action_history = []

    def add_edge(self, u: Node, v: Node, value: Dict):
        """
        Add an edge to the graph
        :param u: u node
        :param v: v node
        :return:
        """

        if u.id not in self.nodes:
            raise GraphError(f"Node {u} does not exist")
        if v.id not in self.nodes:
            raise GraphError(f"Node {v} does not exist")

        if u.id == v.id:
            raise GraphError("Self loops are currently not allowed")

        if u.id not in self.edges:
            self.edges[u.id] = {}
            self._reverse_edges[u.id] = {}
        if v.id not in self.edges:
            self.edges[v.id] = {}
            self._reverse_edges[v.id] = {}

        self.edges[u.id][v.id] = value
        self.edges[v.id][u.id] = value

        self._reverse_edges[u.id][v.id] = value
        self._reverse_edges[v.id][u.id] = value

        self.edge_history[(u.id, v.id)] = []
        self.edge_history[(v.id, u.id)] = []

    def add_directed_edge(self, u: Node, v: Node, value: Dict):
        """
        Add a directed edge from u to v to the graph
        :param u: u node
        :param v: v node
        :return:
        """

        if u.id not in self.nodes:
            raise GraphError(f"Node {u} does not exist")
        if v.id not in self.nodes:
            raise GraphError(f"Node {v} does not exist")

        if u.id == v.id:
            raise GraphError("Self loops are currently not allowed")

        if u.id not in self.edges:
            self.edges[u.id] = {}
        if v.id not in self._reverse_edges:
            self._reverse_edges[v.id] = {}

        self.edges[u.id][v.id] = value
        self._reverse_edges[v.id][u.id] = value

        self.edge_history[(u.id, v.id)] = []

    def retrieve_edge_history(
        self, u, v, action: TestResultAction = None
    ) -> List[TestResult]:
        """
        Retrieve the edge history
        :param u:
        :param v:
        :param action:
        :return:
        """
        if action is None:
            return self.edge_history[(u.id, v.id)]

        if (u.id, v.id) not in self.edge_history:
            return []

        return [i for i in self.edge_history[(u.id, v.id)] if i.action == action]

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

    def remove_edge(self, u: Node, v: Node):
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

        if u.id in self.edges and v.id in self.edges[u.id]:
            del self.edges[u.id][v.id]
            del self._reverse_edges[u.id][v.id]

        if v.id in self.edges and u.id in self.edges[v.id]:
            del self.edges[v.id][u.id]
            del self._reverse_edges[v.id][u.id]

    def remove_directed_edge(self, u: Node, v: Node):
        """
        Remove an edge from the graph
        :param u: u node
        :param v: v node
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

        del self.edges[u.id][v.id]
        del self._reverse_edges[v.id][u.id]

    def update_edge(self, u: Node, v: Node, value: Dict):
        """
        Update an undirected edge in the graph
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
        if v.id not in self.edges:
            raise GraphError(f"Node {v} does not have any edges")

        if u.id not in self.edges[v.id]:
            raise GraphError(f"There is no edge from {u} to {v}")

        if v.id not in self.edges[u.id]:
            raise GraphError(f"There is no edge from {v} to {u}")

        self.edges[u.id][v.id] = value
        self.edges[v.id][u.id] = value

        self._reverse_edges[u.id][v.id] = value
        self._reverse_edges[v.id][u.id] = value

    def update_directed_edge(self, u: Node, v: Node, value: Dict):
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

        self.edges[u.id][v.id] = value
        self._reverse_edges[v.id][u.id] = value

    def edge_exists(self, u: Node, v: Node):
        """
        Check if any edge exists between u and v. Cases: u -> v, u <-> v, u <- v
        :param u: node u
        :param v: node v
        :return: True if any edge exists, False otherwise
        """
        if u.id not in self.nodes:
            return False
        if v.id not in self.nodes:
            return False
        if u.id in self.edges and v.id in self.edges[u.id]:
            return True
        if v.id in self.edges and u.id in self.edges[v.id]:
            return True
        return False

    def directed_edge_exists(self, u: Node, v: Node):
        """
        Check if a directed edge exists between u and v. Cases: u -> v, u <-> v
        :param u: node u
        :param v: node v
        :return: True if a directed edge exists, False otherwise
        """
        if u.id not in self.nodes:
            return False
        if v.id not in self.nodes:
            return False
        if u.id not in self.edges:
            return False
        if v.id not in self.edges[u.id]:
            return False
        return True

    def only_directed_edge_exists(self, u: Node, v: Node):
        """
        Check if a directed edge exists between u and v, but no directed edge exists between v and u. Case: u -> v
        :param u: node u
        :param v: node v
        :return: True if only directed edge exists, False otherwise
        """
        if self.directed_edge_exists(u, v) and not self.directed_edge_exists(v, u):
            return True
        return False

    def undirected_edge_exists(self, u: Node, v: Node):
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

    def bidirected_edge_exists(self, u: Node, v: Node):
        """
        Check if a bidirected edge exists between u and v. Note: currently, a bidirected edges is implemented just as
        an undirected edge. However, they are two functions as they mean different things in different algorithms.
        This function will be used for the FCI algorithm for now, where a bidirected edge is an edge between two nodes
        that have been identified to have a common cause by orientation rules.
        Later, a cohersive naming scheme should be implemented.
        :param u: node u
        :param v: node v
        :return: True if a bidirected edge exists, False otherwise
        """
        if self.directed_edge_exists(u, v) and self.directed_edge_exists(v, u):
            return True
        return False

    def edge_value(self, u: Node, v: Node) -> Optional[Dict]:
        """
        retrieve the value of an edge
        :param u:
        :param v:
        :return:
        """

        if u.id not in self.edges:
            return None
        if v.id not in self.edges[u.id]:
            return None

        return self.edges[u.id][v.id]

    def add_node(
        self,
        name: str,
        values: Union[List[float], torch.Tensor],
        id_: str = None,
        metadata: Dict[str, any] = None,
    ) -> Node:
        """
        Add a node to the graph
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
                tensor_values = torch.tensor(values, dtype=torch.float32)
            except TypeError as e:
                raise ValueError(f"Currently only numeric values are supported. {e}")

        if metadata is None:
            metadata = {}

        node = Node(name=name, id=id_, values=tensor_values, metadata=metadata)

        self.nodes[id_] = node
        return node

    def directed_path_exists(self, u: Node, v: Node):
        """
        Check if a directed path from u to v exists
        :param u: node u
        :param v: node v
        :return: True if a directed path exists, False otherwise
        """
        if self.directed_edge_exists(u, v):
            return True
        for w in self.edges[u.id]:
            if self.directed_path_exists(self.nodes[w], v):
                return True
        return False

    def directed_paths(self, u: Node, v: Node):
        """
        Return all directed paths from u to v
        :param u: node u
        :param v: node v
        :return: list of directed paths
        """
        # TODO: try a better data structure for this
        if self.directed_edge_exists(u, v):
            return [[(u, v)]]
        paths = []
        for w in self.edges[u.id]:
            if self.directed_edge_exists(u, self.nodes[w]):
                for path in self.directed_paths(self.nodes[w], v):
                    paths.append([(u, self.nodes[w])] + path)
        return paths

    def parents_of_node(self, u: Node):
        """
        Return all parents of a node u
        :param u: node u
        :return: list of nodes (parents)
        """
        return [self.nodes[n] for n in self._reverse_edges[u.id].keys()]

    def inducing_path_exists(self, u: Node, v: Node):
        """
        Check if an inducing path from u to v exists.
        An inducing path from u to v is a directed path from u to v on which all mediators are colliders.
        :param u: node u
        :param v: node v
        :return: True if an inducing path exists, False otherwise
        """
        if not self.directed_path_exists(u, v):
            return False
        for path in self.directed_paths(u, v):
            for i in range(1, len(path) - 1):
                r, w = path[i]
                if not self.bidirected_edge_exists(r, w):
                    # TODO: check if this is correct (@sof)
                    return True
        return False
