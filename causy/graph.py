import importlib
import itertools
import json
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Dict, Set
import multiprocessing as mp

from causy.independence_tests import (
    CorrelationCoefficientTest,
    IndependenceTestInterface,
    PartialCorrelationTest,
    CalculateCorrelations,
    ExtendedPartialCorrelationTestMatrix,
)

from causy.orientation_tests import ColliderTest

from causy.interfaces import (
    BaseGraphInterface,
    NodeInterface,
    TestResultAction,
    ComparisonSettings,
    AS_MANY_AS_FIELDS,
    TestResult,
    GraphModelInterface,
    LogicStepInterface,
    GeneratorInterface,
)

import logging

logger = logging.getLogger(__name__)

DEFAULT_INDEPENDENCE_TEST = CorrelationCoefficientTest


@dataclass
class Node(NodeInterface):
    name: str
    values: List[float]

    def __hash__(self):
        return hash(self.name)


class UndirectedGraphError(Exception):
    pass


class UndirectedGraph(BaseGraphInterface):
    nodes: Dict[str, Node]
    edges: Dict[Node, Dict[Node, Dict]]
    edge_history: Dict[Set[Node], List[TestResult]]
    action_history: List[Dict[str, List[TestResult]]]

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.edge_history = {}

    def add_edge(self, u: Node, v: Node, value: Dict):
        """
        Add an edge to the graph
        :param u: u node
        :param v: v node
        :return:
        """
        if u.name not in self.nodes:
            raise UndirectedGraphError(f"Node {u} does not exist")
        if v.name not in self.nodes:
            raise UndirectedGraphError(f"Node {v} does not exist")
        if u not in self.edges:
            self.edges[u] = {}
        if v not in self.edges:
            self.edges[v] = {}

        self.edges[u][v] = value
        self.edges[v][u] = value

        self.edge_history[(u, v)] = []
        self.edge_history[(v, u)] = []

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
            return self.edge_history[(u, v)]

        return [i for i in self.edge_history[(u, v)] if i.action == action]

    def add_edge_history(self, u, v, action: TestResultAction):
        if (u, v) not in self.edge_history:
            self.edge_history[(u, v)] = []
        self.edge_history[(u, v)].append(action)

    def remove_edge(self, u: Node, v: Node):
        """
        Remove an edge from the graph
        :param u: u node
        :param v: v node
        :return:
        """
        if u.name not in self.nodes:
            raise UndirectedGraphError(f"Node {u} does not exist")
        if v.name not in self.nodes:
            raise UndirectedGraphError(f"Node {v} does not exist")
        if u not in self.edges:
            raise UndirectedGraphError(f"Node {u} does not have any nodes")
        if v not in self.edges:
            raise UndirectedGraphError(f"Node {v} does not have any nodes")

        if v not in self.edges[u]:
            return
        del self.edges[u][v]

        if u not in self.edges[v]:
            return
        del self.edges[v][u]

    def remove_directed_edge(self, u: Node, v: Node):
        """
        Remove an edge from the graph
        :param u: u node
        :param v: v node
        :return:
        """
        if u.name not in self.nodes:
            raise UndirectedGraphError(f"Node {u} does not exist")
        if v.name not in self.nodes:
            raise UndirectedGraphError(f"Node {v} does not exist")
        if u not in self.edges:
            raise UndirectedGraphError(f"Node {u} does not have any nodes")
        if v not in self.edges:
            raise UndirectedGraphError(f"Node {v} does not have any nodes")

        if v not in self.edges[u]:
            return
        del self.edges[u][v]

    def update_edge(self, u: Node, v: Node, value: Dict):
        """
        Update an edge in the graph
        :param u: u node
        :param v: v node
        :return:
        """
        if u.name not in self.nodes:
            raise UndirectedGraphError(f"Node {u} does not exist")
        if v.name not in self.nodes:
            raise UndirectedGraphError(f"Node {v} does not exist")
        if u not in self.edges:
            raise UndirectedGraphError(f"Node {u} does not have any nodes")
        if v not in self.edges:
            raise UndirectedGraphError(f"Node {v} does not have any nodes")

        self.edges[u][v] = value
        self.edges[v][u] = value

    def undirected_edge_exists(self, u: Node, v: Node):
        if u.name not in self.nodes:
            return False
        if v.name not in self.nodes:
            return False
        if u not in self.edges:
            return False
        if v not in self.edges:
            return False
        if u not in self.edges[v]:
            return False
        if v not in self.edges[u]:
            return False
        return True

    def directed_edge_exists(self, u: Node, v: Node):
        if u.name not in self.nodes:
            return False
        if v.name not in self.nodes:
            return False
        if u not in self.edges:
            return False
        if v not in self.edges[u]:
            return False
        return True

    def only_directed_edge_exists(self, u: Node, v: Node):
        if self.directed_edge_exists(u, v) and not self.directed_edge_exists(v, u):
            return True
        return False

    def undirected_edge_exist(self, u: Node, v: Node):
        if self.directed_edge_exists(u, v) and self.directed_edge_exists(v, u):
            return True
        return False

    def edge_value(self, u: Node, v: Node):
        return self.edges[u][v]

    def add_node(self, name: str, values: List[float]):
        """
        Add a node to the graph
        :param name: name of the node
        :param values: values of the node
        :param : node

        :return:
        """
        self.nodes[name] = Node(name, values)


def unpack_run(args):
    tst = args[0]
    del args[0]
    return tst(*args)


class AbstractGraphModel(GraphModelInterface, ABC):
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

        graph = UndirectedGraph()
        for key in keys:
            graph.add_node(key, nodes[key])

        self.graph = graph
        return graph

    def create_all_possible_edges(self):
        """
        Create all possible nodes
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
        :return:
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

        self.graph.action_history = action_history

    def _format_yield(self, test_fn, graph, generator):
        for i in generator:
            yield [test_fn, [*i], graph]

    def _take_action(self, results):
        actions_taken = []
        for result_items in results:
            if result_items is None:
                continue
            if not isinstance(result_items, list):
                result_items = [result_items]

            for i in result_items:
                if i.x is not None and i.y is not None:
                    logger.info(f"Action: {i.action} on {i.x.name} and {i.y.name}")

                # add the action to the actions history
                actions_taken.append(i)

                # execute the action returned by the test
                if i.action == TestResultAction.REMOVE_EDGE_UNDIRECTED:
                    self.graph.remove_edge(i.x, i.y)
                    self.graph.add_edge_history(i.x, i.y, i)
                    self.graph.add_edge_history(i.y, i.x, i)
                elif i.action == TestResultAction.UPDATE_EDGE:
                    self.graph.update_edge(i.x, i.y, i.data)
                    self.graph.add_edge_history(i.x, i.y, i)
                    self.graph.add_edge_history(i.y, i.x, i)
                elif i.action == TestResultAction.DO_NOTHING:
                    pass
                elif i.action == TestResultAction.REMOVE_EDGE_DIRECTED:
                    self.graph.remove_directed_edge(i.x, i.y)
                    self.graph.add_edge_history(i.x, i.y, i)

        return actions_taken

    def execute_pipeline_step(self, test_fn: IndependenceTestInterface):
        """
        Filter the graph
        :param test_fn: the test function
        :param threshold: the threshold
        :return:
        """
        actions_taken = []

        # initialize the worker pool (we currently use all available cores * 2)

        # run all combinations in parallel except if the number of combinations is smaller then the chunk size
        # because then we would create more overhead then we would definetly gain from parallel processing
        if test_fn.PARALLEL:
            for result in self.pool.imap_unordered(
                unpack_run,
                self._format_yield(
                    test_fn, self.graph, test_fn.GENERATOR.generate(self.graph, self)
                ),
                chunksize=test_fn.CHUNK_SIZE_PARALLEL_PROCESSING,
            ):
                if not isinstance(result, list):
                    result = [result]
                actions_taken.extend(self._take_action(result))
        else:
            iterator = [
                unpack_run(i)
                for i in [
                    [test_fn, [*i], self.graph]
                    for i in test_fn.GENERATOR.generate(self.graph, self)
                ]
            ]
            actions_taken.extend(self._take_action(iterator))

        return actions_taken


def graph_model_factory(
    pipeline_steps: Optional[List[IndependenceTestInterface]] = None,
) -> type[AbstractGraphModel]:
    """
    Create a graph model
    :param pipeline_steps: a list of pipeline_steps which should be applied to the graph
    :return: the graph model
    """

    class GraphModel(AbstractGraphModel):
        def __init__(self):
            super().__init__(pipeline_steps=pipeline_steps)

    return GraphModel


class Loop(LogicStepInterface):
    def execute(
        self, graph: BaseGraphInterface, graph_model_instance_: GraphModelInterface
    ):
        n = 0
        while not self.exit_condition((graph, n)):
            for pipeline_step in self.pipeline_steps:
                graph_model_instance_.execute_pipeline_step(pipeline_step)
            n += 1

    def __init__(
        self,
        pipeline_steps: Optional[List[IndependenceTestInterface]] = None,
        exit_condition=None,
    ):
        super().__init__()
        self.pipeline_steps = pipeline_steps or []
        self.exit_condition = exit_condition


PCGraph = graph_model_factory(
    pipeline_steps=[
        CalculateCorrelations(),
        CorrelationCoefficientTest(threshold=0.1),
        PartialCorrelationTest(threshold=0.1),
        ExtendedPartialCorrelationTestMatrix(threshold=0.1),
        ColliderTest(),
        # check replacing it with a loop of ExtendedPartialCorrelationTest
        # Loop(
        #    pipeline_steps=[
        #        PlaceholderTest(),
        #    ],
        #    exit_condition=lambda inputs: True if inputs[1] > 2 else False
        # )
    ]
)
