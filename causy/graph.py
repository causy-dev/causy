from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple, Union
from uuid import uuid4
import logging

import torch
import torch.multiprocessing as mp

from causy.interfaces import (
    IndependenceTestInterface,
)
from causy.interfaces import (
    BaseGraphInterface,
    NodeInterface,
    TestResultAction,
    TestResult,
    GraphModelInterface,
    LogicStepInterface,
    ExitConditionInterface,
)
from causy.utils import (
    load_pipeline_artefact_by_definition,
    load_pipeline_steps_by_definition,
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


def unpack_run(args):
    tst = args[0]
    del args[0]
    return tst(*args)


class AbstractGraphModel(GraphModelInterface, ABC):
    """
    The graph model is the main class of causy. It is responsible for creating a graph from data and executing the pipeline_steps.

    The graph model is responsible for the following tasks:
    - Create a graph from data (create_graph_from_data)
    - Execute the pipeline_steps (execute_pipeline_steps)
    - Take actions on the graph (execute_pipeline_step & _take_action which is called by execute_pipeline_step)

    It also initializes and takes care of the multiprocessing pool.

    """

    pipeline_steps: List[IndependenceTestInterface]
    graph: BaseGraphInterface
    pool: mp.Pool

    def __init__(
        self,
        graph=None,
        pipeline_steps: Optional[List[IndependenceTestInterface]] = None,
    ):
        self.graph = graph
        self.pipeline_steps = pipeline_steps or []
        self.pool = mp.Pool(mp.cpu_count() * 2)

    def create_graph_from_data(self, data: List[Dict[str, float]]):
        """
        Create a graph from data
        :param data: is a list of dictionaries
        :return:
        """
        # initialize nodes
        keys = data[0].keys()
        nodes: Dict[str, List[float]] = {}

        for key in keys:
            nodes[key] = []

        # load nodes into node dict
        for row in data:
            for key in keys:
                nodes[key].append(row[key])

        graph = Graph()
        for key in keys:
            graph.add_node(key, nodes[key])

        self.graph = graph
        return graph

    def create_all_possible_edges(self):
        """
        Create all possible edges on a graph
        TODO: replace me with the skeleton builders
        :return:
        """
        for u in self.graph.nodes.values():
            for v in self.graph.nodes.values():
                if u == v:
                    continue
                self.graph.add_edge(u, v, {})

    def execute_pipeline_steps(self):
        """
        Execute all pipeline_steps
        :return: the steps taken during the step execution
        """
        action_history = []

        for filter in self.pipeline_steps:
            logger.info(f"Executing pipeline step {filter.__class__.__name__}")
            if isinstance(filter, LogicStepInterface):
                filter.execute(self.graph, self)
                continue

            result = self.execute_pipeline_step(filter)
            action_history.append(
                {"step": filter.__class__.__name__, "actions": result}
            )

        return action_history

    def _format_yield(self, test_fn, graph, generator):
        """
        Format the yield for the parallel processing
        :param test_fn: the pipeline_step test function
        :param graph: the graph
        :param generator: the generator object which generates the combinations
        :return: yields the test function with its inputs
        """
        for i in generator:
            yield [test_fn, [*i], graph]

    def _take_action(self, results):
        """
        Take the actions returned by the test

        In causy changes on the graph are not executed directly. Instead, the test returns an action which should be executed on the graph.
        This is done to make it possible to execute the tests in parallel as well as to decide proactively at which point in the decisions taken by the pipeline step should be executed.
        Actions are returned by the test and are executed on the graph. The actions are stored in the action history to make it possible to revert the actions or use them in a later step.

        :param results:
        :return:
        """
        actions_taken = []
        for result_items in results:
            if result_items is None:
                continue
            if not isinstance(result_items, list):
                result_items = [result_items]

            for i in result_items:
                if i.x is not None and i.y is not None:
                    logger.info(f"Action: {i.action} on {i.x.name} and {i.y.name}")

                # execute the action returned by the test
                if i.action == TestResultAction.REMOVE_EDGE_UNDIRECTED:
                    if not self.graph.undirected_edge_exists(i.x, i.y):
                        logger.debug(
                            f"Tried to remove undirected edge {i.x.name} <-> {i.y.name}. But it does not exist."
                        )
                        continue
                    self.graph.remove_edge(i.x, i.y)
                    self.graph.add_edge_history(i.x, i.y, i)
                    self.graph.add_edge_history(i.y, i.x, i)
                elif i.action == TestResultAction.UPDATE_EDGE:
                    if not self.graph.edge_exists(i.x, i.y):
                        logger.debug(
                            f"Tried to update edge {i.x.name} -> {i.y.name}. But it does not exist."
                        )
                        continue
                    self.graph.update_edge(i.x, i.y, i.data)
                    self.graph.add_edge_history(i.x, i.y, i)
                    self.graph.add_edge_history(i.y, i.x, i)
                elif i.action == TestResultAction.UPDATE_EDGE_DIRECTED:
                    if not self.graph.directed_edge_exists(i.x, i.y):
                        logger.debug(
                            f"Tried to update directed edge {i.x.name} -> {i.y.name}. But it does not exist."
                        )
                        continue
                    self.graph.update_directed_edge(i.x, i.y, i.data)
                    self.graph.add_edge_history(i.x, i.y, i)
                elif i.action == TestResultAction.DO_NOTHING:
                    continue
                elif i.action == TestResultAction.REMOVE_EDGE_DIRECTED:
                    if not self.graph.directed_edge_exists(i.x, i.y):
                        logger.debug(
                            f"Tried to remove directed edge {i.x.name} -> {i.y.name}. But it does not exist."
                        )
                        continue
                    self.graph.remove_directed_edge(i.x, i.y)
                    self.graph.add_edge_history(i.x, i.y, i)

                # add the action to the actions history
                actions_taken.append(i)

        return actions_taken

    def execute_pipeline_step(self, test_fn: IndependenceTestInterface):
        """
        Execute a single pipeline_step on the graph. either in parallel or in a single process depending on the test_fn.parallel flag
        :param test_fn: the test function
        :param threshold: the threshold
        :return:
        """
        actions_taken = []

        # initialize the worker pool (we currently use all available cores * 2)

        # run all combinations in parallel except if the number of combinations is smaller then the chunk size
        # because then we would create more overhead then we would definetly gain from parallel processing
        if test_fn.parallel:
            for result in self.pool.imap_unordered(
                unpack_run,
                self._format_yield(
                    test_fn, self.graph, test_fn.generator.generate(self.graph, self)
                ),
                chunksize=test_fn.chunk_size_parallel_processing,
            ):
                if not isinstance(result, list):
                    result = [result]
                actions_taken.extend(self._take_action(result))
        else:
            if test_fn.generator.chunked:
                for chunk in test_fn.generator.generate(self.graph, self):
                    iterator = [
                        unpack_run(i)
                        for i in [[test_fn, [*c], self.graph] for c in chunk]
                    ]
                    actions_taken.extend(self._take_action(iterator))
            else:
                iterator = [
                    unpack_run(i)
                    for i in [
                        [test_fn, [*i], self.graph]
                        for i in test_fn.generator.generate(self.graph, self)
                    ]
                ]
                actions_taken.extend(self._take_action(iterator))

        self.graph.action_history.append(
            {"step": type(test_fn).__name__, "actions": actions_taken}
        )

        return actions_taken


def graph_model_factory(
    pipeline_steps: Optional[List[IndependenceTestInterface]] = None,
) -> type[AbstractGraphModel]:
    """
    Create a graph model based on a List of pipeline_steps
    :param pipeline_steps: a list of pipeline_steps which should be applied to the graph
    :return: the graph model
    """

    class GraphModel(AbstractGraphModel):
        def __init__(self):
            super().__init__(pipeline_steps=pipeline_steps)

    return GraphModel


class Loop(LogicStepInterface):
    """
    A loop which executes a list of pipeline_steps until the exit_condition is met.
    """

    def execute(
        self, graph: BaseGraphInterface, graph_model_instance_: GraphModelInterface
    ):
        """
        Executes the loop til self.exit_condition is met
        :param graph:
        :param graph_model_instance_:
        :return:
        """
        n = 0
        steps = None
        while not self.exit_condition(
            graph=graph,
            graph_model_instance_=graph_model_instance_,
            actions_taken=steps,
            iteration=n,
        ):
            steps = []
            for pipeline_step in self.pipeline_steps:
                result = graph_model_instance_.execute_pipeline_step(pipeline_step)
                steps.extend(result)
            n += 1

    def __init__(
        self,
        pipeline_steps: Optional[List[IndependenceTestInterface]] = None,
        exit_condition: ExitConditionInterface = None,
    ):
        super().__init__()
        # TODO check if this is a good idea
        if isinstance(exit_condition, dict):
            exit_condition = load_pipeline_artefact_by_definition(exit_condition)

        # TODO: check if this is a good idea
        if len(pipeline_steps) > 0 and isinstance(pipeline_steps[0], dict):
            pipeline_steps = load_pipeline_steps_by_definition(pipeline_steps)

        self.pipeline_steps = pipeline_steps or []
        self.exit_condition = exit_condition
