import asyncio
import logging
from abc import ABC
from typing import Optional, List, Dict

import torch.multiprocessing as mp

from causy.graph import Graph
from causy.graph_utils import unpack_run, generate_to_queue, collect_and_execute_tests
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

        self.pool = mp.Pool(mp.cpu_count() * 2)

        for filter in self.pipeline_steps:
            logger.info(f"Executing pipeline step {filter.__class__.__name__}")
            if isinstance(filter, LogicStepInterface):
                filter.execute(self.graph, self)
                continue

            result = self.execute_pipeline_step(filter)
            action_history.append(
                {"step": filter.__class__.__name__, "actions": result}
            )

        self.pool.close()

        return action_history

    def _format_yield(self, test_fn, graph, generator, test_queue, result_queue):
        """
        Format the yield for the parallel processing
        :param test_fn: the pipeline_step test function
        :param graph: the graph
        :param generator: the generator object which generates the combinations
        :param test_queue: the queue to which the tests should be written
        :param result_queue: the queue to which the results should be written
        :return: yields the test function with its inputs
        """

        print("generator.ready()", generator.ready())
        print("test_queue.qsize()", test_queue.qsize())
        while generator.ready() is False or test_queue.qsize() > 0:
            if test_queue.qsize() > 0:
                try:
                    yield [test_fn, [*test_queue.get()], graph, result_queue]
                except Exception as e:
                    print("Exception")
                    print(e)
                    continue

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

    def execute_pipeline_step(self, test_fn: PipelineStepInterface):
        """
        Execute a single pipeline_step on the graph. either in parallel or in a single process depending on the test_fn.parallel flag
        :param test_fn: the test function
        :param threshold: the threshold
        :return:
        """
        actions_taken = []

        m = mp.Manager()
        result_queue = m.Queue()
        test_queue = m.Queue()

        # initialize the worker pool (we currently use all available cores * 2)

        # run all combinations in parallel except if the number of combinations is smaller than the chunk size
        # because then we would create more overhead than we would definitely gain from parallel processing

        # print(generate_to_queue(test_fn.generator, self.graph, self, test_queue))

        generator_process = self.pool.apply_async(
            generate_to_queue,
            (
                test_fn.generator,
                self.graph,
                test_queue,
            ),
        )

        tests_to_execute = []

        while (
            result_queue.qsize() > 0
            or generator_process.ready() is False
            or test_queue.qsize() > 0
        ):
            if result_queue.qsize() > 0:
                try:
                    result = result_queue.get(False)
                except Exception as e:
                    print("Exception")
                    print(e)
                    continue
                self._take_action([result])
                actions_taken.append(result)

            if generator_process.ready() is False or test_queue.qsize() > 0:
                tests_to_execute.append(test_queue.get())

            if len(tests_to_execute) == test_fn.chunk_size_parallel_processing * 10:
                progress = self.pool.map_async(
                    unpack_run,
                    [
                        [test_fn, test, self.graph, result_queue]
                        for test in tests_to_execute
                    ],
                    chunksize=test_fn.chunk_size_parallel_processing,
                )
                tests_to_execute = []

            if (
                generator_process.ready() is True
                and test_queue.qsize() > 0
                and len(tests_to_execute) == 0
            ):
                progress = self.pool.map_async(
                    unpack_run,
                    [
                        (test_fn, test, self.graph, result_queue)
                        for test in tests_to_execute
                    ],
                    chunksize=test_fn.chunk_size_parallel_processing,
                )
                tests_to_execute = []

        logger.debug(
            f"Finished pipeline step {test_fn.__class__.__name__} with {len(actions_taken)} actions"
        )

        print("Ready: " + str(generator_process.ready()))

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
