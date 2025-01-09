import inspect
import logging
import platform
from abc import ABC
from copy import deepcopy
import time
from typing import Optional, List, Dict, Callable, Union, Any, Generator, Tuple

import torch.multiprocessing as mp

from causy.data_loader import AbstractDataLoader
from causy.edge_types import DirectedEdge
from causy.graph import GraphManager, Graph, GraphError
from causy.graph_utils import unpack_run
from causy.interfaces import (
    PipelineStepInterface,
    LogicStepInterface,
    BaseGraphInterface,
    GraphModelInterface,
    GraphUpdateHook,
)
from causy.models import TestResultAction, Algorithm, ActionHistoryStep
from causy.variables import (
    resolve_variables_to_algorithm_for_pipeline_steps,
    resolve_variables,
    VariableType,
    VariableTypes,
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

    algorithm: Algorithm
    pipeline_steps: List[PipelineStepInterface]
    graph: BaseGraphInterface
    pool: mp.Pool

    _graph_class: type[BaseGraphInterface]

    def __init__(
        self,
        graph=None,
        algorithm: Algorithm = None,
    ):
        self.graph = graph
        self.algorithm = algorithm

        if algorithm.pipeline_steps is not None:
            self.pipeline_steps = algorithm.pipeline_steps or []

        if self.__multiprocessing_required(self.pipeline_steps):
            self.pool = self.__initialize_pool()
        else:
            self.pool = None

    def __initialize_pool(self) -> mp.Pool:
        """
        Initialize the multiprocessing pool
        :return: the multiprocessing pool
        """
        # we need to set the start method to spawn because the default fork method does not work well with torch on linux
        # see https://pytorch.org/docs/stable/notes/multiprocessing.html
        if platform.system() == "Linux":
            try:
                mp.set_start_method("spawn")
            except RuntimeError:
                logger.warning(
                    "Could not set multiprocessing start method to spawn. Using default method. This might cause issues on Linux."
                )
        return mp.Pool(mp.cpu_count() * 2)

    def __multiprocessing_required(self, pipeline_steps):
        """
        Check if multiprocessing is required
        :param pipeline_steps: the pipeline steps
        :return: True if multiprocessing is required
        """
        for step in pipeline_steps:
            if hasattr(step, "parallel") and step.parallel:
                return True
        return False

    def __del__(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def __create_graph_from_dict(self, data: Dict[str, List[float]]):
        """
        Create a graph from a dictionary
        :param data: the dictionary
        :return: the graph
        """
        graph = GraphManager(graph_class=self._graph_class)
        for key, values in sorted(data.items()):
            graph.add_node(key, values, id_=key)
        return graph

    def __create_graph_from_list(self, data: List[Dict[str, float]]):
        """
        Create a graph from a list of dictionaries
        :param data:
        :return:
        """
        # initialize nodes
        keys = data[0].keys()
        nodes: Dict[str, List[float]] = {}

        for key in sorted(keys):
            nodes[key] = []

        # load nodes into node dict
        for row in data:
            for key in keys:
                nodes[key].append(row[key])

        graph = GraphManager(graph_class=self._graph_class)
        for key in keys:
            graph.add_node(key, nodes[key], id_=key)

        return graph

    def _create_from_data_loader(self, data_loader: AbstractDataLoader):
        """
        Create a graph from a data loader
        :param data_loader: the data loader
        :return: the graph
        """
        nodes: Dict[str, List[float]] = {}
        keys = None

        # load nodes into node dict
        for row in data_loader.load():
            if isinstance(row, dict) and "_dict" in row:
                # edge case for when data is in a dict of lists
                return self.__create_graph_from_dict(row["_dict"])

            if keys is None:
                keys = row.keys()
                for key in sorted(keys):
                    nodes[key] = []

            for key in keys:
                nodes[key].append(row[key])

        graph = GraphManager(graph_class=self._graph_class)
        for key in keys:
            graph.add_node(key, nodes[key], id_=key)

        return graph

    def create_graph_from_data(
        self,
        data: Union[List[Dict[str, float]], Dict[str, List[float]], AbstractDataLoader],
    ):
        """
        Create a graph from data
        :param data: is a list of dictionaries or a dictionary with lists
        :return:
        """

        if isinstance(data, AbstractDataLoader):
            graph = self._create_from_data_loader(data)
        elif isinstance(data, dict):
            graph = self.__create_graph_from_dict(data)
        else:
            graph = self.__create_graph_from_list(data)

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

    def execute_pipeline_steps(self) -> List[ActionHistoryStep]:
        """
        Execute all pipeline_steps
        :return: the steps taken during the step execution
        """
        all(self.execute_pipeline_step_with_progress())
        return self.graph.action_history

    def execute_pipeline_step_with_progress(self) -> Generator:
        started = time.time()
        for filter in self.pipeline_steps:
            logger.info(f"Executing pipeline step {filter.__class__.__name__}")
            if isinstance(filter, LogicStepInterface):
                actions_taken = filter.execute(self.graph.graph, self)
                self.graph.graph.action_history.append(actions_taken)
                continue
            yield {
                "step": filter.__class__.__name__,
                "previous_duration": time.time() - started,
            }
            started = time.time()
            actions_taken, all_actions = self.execute_pipeline_step(filter)
            self.graph.graph.action_history.append(
                ActionHistoryStep(
                    name=filter.name,
                    actions=actions_taken,
                    all_proposed_actions=all_actions,
                    duration=time.time() - started,
                )
            )

            self.graph.purge_soft_deleted_edges()

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

    def _execute_pre_graph_update_hooks(
        self,
        hooks: List[GraphUpdateHook],
        graph: BaseGraphInterface,
        results: List[TestResultAction],
    ) -> List[TestResultAction]:
        """
        Execute the graph update hooks
        :param graph: the graph
        :param hooks: the hooks
        :param results: the results
        :return:
        """
        if not hooks:
            return results
        for hook in hooks:
            results = hook.execute(graph, results)
        return results

    def _execute_post_graph_update_hooks(
        self,
        hooks: List[GraphUpdateHook],
        graph: BaseGraphInterface,
        results: List[TestResultAction],
    ) -> None:
        """
        Execute the graph update hooks
        :param graph: the graph
        :param hooks: the hooks
        :param results: the results
        :return:
        """
        if not hooks:
            return
        for hook in hooks:
            hook.execute(graph, results)

    def _execute_hook(self, hook: str, i: TestResultAction):
        """
        Execute a single hook
        :param hook: the name of the hook
        :param i: the test result action
        :return:
        """

        try:
            getattr(self.graph.get_edge(i.u, i.v).edge_type, hook)(self.graph, i)
        except GraphError as e:
            logger.info(f"Graph Error in hook ({str(hook)}): {e}")
        except Exception as e:
            logger.error(f"Error in hook ({str(hook)}): {e}")

    def _take_action(
        self, results, dry_run=False
    ) -> Tuple[List[TestResultAction], List[TestResultAction]]:
        """
        Take the actions returned by the test

        In causy changes on the graph are not executed directly. Instead, the test returns an action which should be executed on the graph.
        This is done to make it possible to execute the tests in parallel as well as to decide proactively at which point in the decisions taken by the pipeline step should be executed.
        Actions are returned by the test and are executed on the graph. The actions are stored in the action history to make it possible to revert the actions or use them in a later step.

        :param results: the results

        :return: actions taken, all actions taken
        """
        all_actions_taken = []
        all_actions = []
        for result_items in results:
            if result_items is None:
                continue
            if not isinstance(result_items, list):
                result_items = [result_items]
            result_items = self._execute_pre_graph_update_hooks(
                self.algorithm.pre_graph_update_hooks, self.graph, result_items
            )
            actions_taken = []
            all_actions.extend(result_items)
            for i in result_items:
                if i.u is not None and i.v is not None:
                    logger.debug(f"Action: {i.action} on {i.u.name} and {i.v.name}")

                if dry_run:
                    actions_taken.append(i)
                    continue

                # execute the action returned by the test
                if i.action == TestResultAction.REMOVE_EDGE_UNDIRECTED:
                    if not self.graph.undirected_edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to remove undirected edge {i.u.name} <-> {i.v.name}. But it does not exist."
                        )
                        continue
                    self._execute_hook("pre_edge_remove_hook", i)
                    self.graph.remove_edge(i.u, i.v, soft_delete=True)
                    self.graph.add_edge_history(i.u, i.v, i)
                    self.graph.add_edge_history(i.v, i.u, i)
                    self._execute_hook("post_edge_remove_hook", i)
                elif i.action == TestResultAction.UPDATE_EDGE:
                    if not self.graph.edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to update edge {i.u.name} -> {i.v.name}. But it does not exist."
                        )
                        continue
                    self._execute_hook("pre_edge_update_hook", i)
                    self.graph.update_edge(i.u, i.v, metadata=i.data)
                    self.graph.add_edge_history(i.u, i.v, i)
                    self.graph.add_edge_history(i.v, i.u, i)
                    self._execute_hook("post_edge_update_hook", i)
                elif i.action == TestResultAction.UPDATE_EDGE_DIRECTED:
                    if not self.graph.directed_edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to update directed edge {i.u.name} -> {i.v.name}. But it does not exist."
                        )
                        continue

                    self._execute_hook("pre_edge_update_hook", i)
                    self.graph.update_directed_edge(i.u, i.v, i.data)
                    self.graph.add_edge_history(i.u, i.v, i)
                    self._execute_hook("post_edge_update_hook", i)
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
                    self._execute_hook("pre_edge_remove_hook", i)
                    self.graph.remove_directed_edge(i.u, i.v, soft_delete=True)
                    # TODO: move this to pre/post update hooks
                    if self.graph.edge_exists(
                        i.v, i.u
                    ):  # if the edge is undirected, make it directed
                        self.graph.update_directed_edge(
                            i.v, i.u, edge_type=DirectedEdge()
                        )
                    self.graph.add_edge_history(i.u, i.v, i)
                    self._execute_hook("post_edge_remove_hook", i)
                elif i.action == TestResultAction.UPDATE_EDGE_TYPE:
                    if not self.graph.edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to update edge type {i.u.name} <-> {i.v.name}. But it does not exist."
                        )
                        continue
                    self._execute_hook("pre_edge_type_change_hook", i)
                    self.graph.update_edge(i.u, i.v, edge_type=i.edge_type)
                    self.graph.add_edge_history(i.u, i.v, i)
                    self.graph.add_edge_history(i.v, i.u, i)
                    self._execute_hook("post_edge_type_change_hook", i)
                elif i.action == TestResultAction.UPDATE_EDGE_TYPE_DIRECTED:
                    if not self.graph.directed_edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to update edge type {i.u.name} -> {i.v.name}. But it does not exist."
                        )
                        continue
                    self._execute_hook("pre_edge_type_change_hook", i)
                    self.graph.update_directed_edge(i.u, i.v, edge_type=i.edge_type)
                    self.graph.add_edge_history(i.u, i.v, i)
                    self._execute_hook("post_edge_type_change_hook", i)

                elif i.action == TestResultAction.RESTORE_EDGE:
                    if self.graph.edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to restore edge {i.u.name} <-> {i.v.name}. But it does exist."
                        )
                        continue

                    self.graph.restore_edge(i.u, i.v)
                    self.graph.add_edge_history(i.u, i.v, i)
                    self.graph.add_edge_history(i.v, i.u, i)

                elif i.action == TestResultAction.RESTORE_EDGE_DIRECTED:
                    if self.graph.directed_edge_exists(i.u, i.v):
                        logger.debug(
                            f"Tried to restore edge {i.u.name} <-> {i.v.name}. But it does exist."
                        )
                        continue

                    self.graph.restore_directed_edge(i.u, i.v)
                    self.graph.add_edge_history(i.u, i.v, i)
                # add the action to the actions history
                actions_taken.append(i)
            all_actions_taken.extend(actions_taken)
            self._execute_post_graph_update_hooks(
                self.algorithm.post_graph_update_hooks, self.graph, actions_taken
            )
        return all_actions_taken, all_actions

    def execute_pipeline_step(
        self, test_fn: PipelineStepInterface, apply_to_graph=True
    ) -> Tuple[List[TestResultAction], List[TestResultAction]]:
        """
        Execute a single pipeline_step on the graph. either in parallel or in a single process depending on the test_fn.parallel flag
        :param apply_to_graph:  if the action should be applied to the graph
        :param test_fn: the test function
        :param threshold: the threshold
        :return:
        """
        actions_taken = []
        all_actions = []
        # initialize the worker pool (we currently use all available cores * 2)

        # run all combinations in parallel except if the number of combinations is smaller then the chunk size
        # because then we would create more overhead then we would definetly gain from parallel processing
        if test_fn.parallel:
            if self.pool is None:
                logger.warning(
                    "Parallel processing is enabled but no pool is initialized. Initializing pool."
                )
                self.pool = self.__initialize_pool()

            for result in self.pool.imap_unordered(
                unpack_run,
                self._format_yield(
                    test_fn,
                    self.graph.graph,
                    test_fn.generator.generate(self.graph.graph, self),
                ),
                chunksize=test_fn.chunk_size_parallel_processing,
            ):
                if not isinstance(result, list):
                    result = [result]

                actions_taken_current, all_actions_current = self._take_action(
                    result, dry_run=not apply_to_graph
                )
                actions_taken.extend(actions_taken_current)
                all_actions.extend(all_actions_current)
        else:
            if test_fn.generator.chunked:
                for chunk in test_fn.generator.generate(self.graph.graph, self):
                    iterator = [
                        unpack_run(i)
                        for i in [[test_fn, [*c], self.graph.graph] for c in chunk]
                    ]
                    actions_taken_current, all_actions_current = self._take_action(
                        iterator, dry_run=not apply_to_graph
                    )
                    actions_taken.extend(actions_taken_current)
                    all_actions.extend(all_actions_current)
            else:
                # this is the only mode which supports unapplied actions to be passed to the next pipeline step (for now)
                # which are sometimes needed for e.g. conflict resolution
                iterator = [
                    i
                    for i in [
                        [test_fn, [*i], self.graph.graph]
                        for i in test_fn.generator.generate(self.graph.graph, self)
                    ]
                ]

                local_results = []
                for i in iterator:
                    rn_fn = i[0]
                    if hasattr(rn_fn, "needs_unapplied_actions"):
                        if rn_fn.needs_unapplied_actions:
                            i.append(local_results)
                    local_results.append(unpack_run(i))
                actions_taken_current, all_actions_current = self._take_action(
                    local_results, dry_run=not apply_to_graph
                )
                actions_taken.extend(actions_taken_current)
                all_actions.extend(all_actions_current)

        return actions_taken, all_actions


def graph_model_factory(
    algorithm: Algorithm = None,
    variables: Dict[str, VariableTypes] = None,
) -> type[AbstractGraphModel]:
    """
    Create a graph model based on a List of pipeline_steps
    :param algorithm: the algorithm which should be used to create the graph model
    :return: the graph model
    """
    original_algorithm = deepcopy(algorithm)
    if variables is None and algorithm.variables is not None:
        variables = resolve_variables(algorithm.variables, {})
    elif variables is None:
        variables = {}

    if len(variables) > 0:
        algorithm.pipeline_steps = resolve_variables_to_algorithm_for_pipeline_steps(
            algorithm.pipeline_steps, variables
        )

    bases = tuple()
    if algorithm.edge_types is not None:
        for edge_type in algorithm.edge_types:
            if not inspect.isabstract(edge_type.GraphAccessMixin):
                bases += (edge_type.GraphAccessMixin,)

    if algorithm.extensions is not None:
        for extension in algorithm.extensions:
            if not inspect.isabstract(extension.GraphAccessMixin):
                bases += (extension.GraphAccessMixin,)

    Graph.__bases__ = tuple(set(Graph.__bases__ + bases))

    class GraphModel(AbstractGraphModel):
        # store the original algorithm for later use like ejecting it without the resolved variables
        _original_algorithm = original_algorithm

        _graph_class = Graph

        def __init__(self):
            super().__init__(algorithm=algorithm)

    return GraphModel
