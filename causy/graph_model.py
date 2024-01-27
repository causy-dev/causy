import logging
from abc import ABC
from copy import deepcopy
from typing import Optional, List, Dict

import torch.multiprocessing as mp

from causy.graph import Graph, EdgeType
from causy.graph_utils import unpack_run
from causy.interfaces import (
    PipelineStepInterface,
    TestResultAction,
    LogicStepInterface,
    BaseGraphInterface,
    GraphModelInterface,
)

logger = logging.getLogger(__name__)


class AbstractGraphModel(GraphModelInterface, ABC):
    """
    The graph model is the main class of causy. It is responsible for creating a graph from data and executing the pipeline_steps.

    The graph model is responsible for the following tasks:
    - Create a graph from data (create_graph_from_data)
    - Execute the pipeline_steps (execute_pipeline_steps)
    - Take actions on the graph (execute_pipeline_step & _take_action which is called by execute_pipeline_step)

    It also initializes and takes care of the multiprocessing pool.

    """

    pipeline_steps: List[PipelineStepInterface]
    graph: BaseGraphInterface
    pool: mp.Pool

    def __init__(
        self,
        graph=None,
        pipeline_steps: Optional[List[PipelineStepInterface]] = None,
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
                if i.u is not None and i.v is not None:
                    logger.info(f"Action: {i.action} on {i.u.name} and {i.v.name}")

                # execute the action returned by the test
                if i.action == TestResultAction.REMOVE_EDGE_UNDIRECTED:
                    if not self.graph.undirected_edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to remove undirected edge {i.u.name} <-> {i.v.name}. But it does not exist."
                        )
                        continue
                    self.graph.remove_edge(i.u, i.v)
                    self.graph.add_edge_history(i.u, i.v, i)
                    self.graph.add_edge_history(i.v, i.u, i)
                elif i.action == TestResultAction.UPDATE_EDGE:
                    if not self.graph.edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to update edge {i.u.name} -> {i.v.name}. But it does not exist."
                        )
                        continue
                    self.graph.update_edge(i.u, i.v, metadata=i.data)
                    self.graph.add_edge_history(i.u, i.v, i)
                    self.graph.add_edge_history(i.v, i.u, i)
                elif i.action == TestResultAction.UPDATE_EDGE_DIRECTED:
                    if not self.graph.directed_edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to update directed edge {i.u.name} -> {i.v.name}. But it does not exist."
                        )
                        continue
                    self.graph.update_directed_edge(i.u, i.v, i.data)
                    self.graph.add_edge_history(i.u, i.v, i)
                elif i.action == TestResultAction.DO_NOTHING:
                    continue
                elif i.action == TestResultAction.REMOVE_EDGE_DIRECTED:
                    if not self.graph.directed_edge_exists(
                        i.u, i.v
                    ) and not self.graph.edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to remove directed edge {i.u.name} -> {i.v.name}. But it does not exist."
                        )
                        continue

                    self.graph.remove_directed_edge(i.u, i.v)
                    # TODO: move this to pre/post update hooks
                    if self.graph.edge_exists(
                        i.v, i.u
                    ):  # if the edge is undirected, make it directed
                        self.graph.update_directed_edge(
                            i.v, i.u, edge_type=EdgeType.DIRECTED
                        )
                    self.graph.add_edge_history(i.u, i.v, i)

                elif i.action == TestResultAction.UPDATE_EDGE_TYPE:
                    if not self.graph.edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to update edge type {i.u.name} <-> {i.v.name}. But it does not exist."
                        )
                        continue
                    self.graph.update_edge(i.u, i.v, edge_type=i.edge_type)
                    self.graph.add_edge_history(i.u, i.v, i)
                    self.graph.add_edge_history(i.v, i.u, i)

                elif i.action == TestResultAction.UPDATE_EDGE_TYPE_DIRECTED:
                    if not self.graph.directed_edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to update edge type {i.u.name} -> {i.v.name}. But it does not exist."
                        )
                        continue
                    self.graph.update_directed_edge(i.u, i.v, edge_type=i.edge_type)
                    self.graph.add_edge_history(i.u, i.v, i)

                # add the action to the actions history
                actions_taken.append(i)

        return actions_taken

    def execute_pipeline_step(self, test_fn: PipelineStepInterface):
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
    pipeline_steps: Optional[List[PipelineStepInterface]] = None,
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
