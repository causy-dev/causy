import enum
from typing import Tuple, List, Optional, Generic
import itertools

from causy.generators import AllCombinationsGenerator
from causy.interfaces import (
    BaseGraphInterface,
    PipelineStepInterface,
    GeneratorInterface,
    PipelineStepInterfaceType,
)
from causy.models import ComparisonSettings, TestResultAction, TestResult
from causy.variables import IntegerParameter, BoolParameter, StringParameter


# theory for all orientation rules with pictures:
# https://hpi.de/fileadmin/user_upload/fachgebiete/plattner/teaching/CausalInference/2019/Introduction_to_Constraint-Based_Causal_Structure_Learning.pdf

# TODO: refactor ColliderTest -> ColliderRule and move to folder orientation_rules (after checking for duplicates)


def filter_unapplied_actions(actions, u, v):
    """
    Filter out actions that have not been applied to the graph yet.
    :param actions: list of actions
    :param u: node u
    :param v: node v
    :return: list of actions that have not been applied to the graph yet
    """
    filtered = []
    for result_set in actions:
        if result_set is None:
            continue
        for result in result_set:
            if result.u == u and result.v == v:
                filtered.append(result)
    return filtered


def generate_restores(unapplied_actions):
    """
    Generate restore actions for unapplied actions.
    :param unapplied_actions: list of unapplied actions
    :param x: node x
    :param y: node y
    :return: list of restore actions
    """
    results = []
    for action in unapplied_actions:
        if action.action == TestResultAction.REMOVE_EDGE_DIRECTED:
            results.append(
                TestResult(
                    u=action.u,
                    v=action.v,
                    action=TestResultAction.RESTORE_EDGE_DIRECTED,
                    data={},
                )
            )
    return results


class ColliderTestConflictResolutionStrategies(enum.StrEnum):
    """
    Enum for the conflict resolution strategies for the ColliderTest.
    """

    # If a conflict occurs, the edge that was removed first is kept.
    KEEP_FIRST = "KEEP_FIRST"

    # If a conflict occurs, the edge that was removed last is kept.
    KEEP_LAST = "KEEP_LAST"


class ColliderTest(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    generator: Optional[GeneratorInterface] = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    chunk_size_parallel_processing: IntegerParameter = 1
    parallel: BoolParameter = False

    conflict_resolution_strategy: StringParameter = (
        ColliderTestConflictResolutionStrategies.KEEP_FIRST
    )

    needs_unapplied_actions: BoolParameter = True

    def process(
        self,
        nodes: Tuple[str],
        graph: BaseGraphInterface,
        unapplied_actions: Optional[List[TestResult]] = None,
    ) -> Optional[List[TestResult] | TestResult]:
        """
        We call triples u, v, z of nodes v structures if u and v that are NOT adjacent but share an adjacent node z.
        V structures looks like this in the undirected skeleton: (u - z - v).
        We now check if z is in the separating set.
        If z is not in the separating set, we know that u and v are uncorrelated given z.
        So, the edges must be oriented from u to z and from v to z (u -> z <- v).
        :param unapplied_actions: list of actions that have not been applied to the graph yet
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """
        # https://github.com/pgmpy/pgmpy/blob/1fe10598df5430295a8fc5cdca85cf2d9e1c4330/pgmpy/estimators/PC.py#L416

        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        # if u and v are adjacent, do nothing
        if graph.undirected_edge_exists(x, y):
            return

        # if u and v are NOT adjacent, store all shared adjacent nodes
        potential_zs = set(graph.edges[x.id].keys()).intersection(
            set(graph.edges[y.id].keys())
        )

        actions = graph.retrieve_edge_history(
            x, y, TestResultAction.REMOVE_EDGE_UNDIRECTED
        )

        # if u and v are not independent given z, safe action: make z a collider
        results = []
        for z in potential_zs:
            z = graph.nodes[z]

            separators = []
            for action in actions:
                if "separatedBy" in action.data:
                    separators += [a.id for a in action.data["separatedBy"]]

            if z.id not in separators:
                unapplied_actions_x_z = filter_unapplied_actions(
                    unapplied_actions, x, z
                )
                unapplied_actions_y_z = filter_unapplied_actions(
                    unapplied_actions, y, z
                )
                if len(unapplied_actions_y_z) > 0 or len(unapplied_actions_x_z) > 0:
                    if (
                        ColliderTestConflictResolutionStrategies.KEEP_FIRST
                        is self.conflict_resolution_strategy
                    ):
                        # We keep the first edge that was removed
                        continue
                    elif (
                        ColliderTestConflictResolutionStrategies.KEEP_LAST
                        is self.conflict_resolution_strategy
                    ):
                        # We keep the last edge that was removed and restore the other edges
                        results.extend(generate_restores(unapplied_actions_x_z))
                        results.extend(generate_restores(unapplied_actions_y_z))
                        results.append(
                            TestResult(
                                u=z,
                                v=x,
                                action=TestResultAction.REMOVE_EDGE_DIRECTED,
                                data={},
                            )
                        )
                        results.append(
                            TestResult(
                                u=z,
                                v=y,
                                action=TestResultAction.REMOVE_EDGE_DIRECTED,
                                data={},
                            )
                        )

                else:
                    results += [
                        TestResult(
                            u=z,
                            v=x,
                            action=TestResultAction.REMOVE_EDGE_DIRECTED,
                            data={},
                        ),
                        TestResult(
                            u=z,
                            v=y,
                            action=TestResultAction.REMOVE_EDGE_DIRECTED,
                            data={},
                        ),
                    ]
        return results


class NonColliderTest(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    generator: Optional[GeneratorInterface] = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    chunk_size_parallel_processing: IntegerParameter = 1
    parallel: BoolParameter = False

    def process(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> Optional[List[TestResult] | TestResult]:
        """
        Further orientation rule: all v structures that are colliders are already oriented.
        We now orient all v structures that have a single alternative to being a collider.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """

        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        # if u and v are adjacent, do nothing
        if graph.edge_exists(x, y):
            return

        # if u and v are NOT adjacent, store all shared adjacent nodes
        potential_zs = set(graph.edges[x.id].keys()).intersection(
            set(graph.edges[y.id].keys())
        )
        # if one edge has an arrowhead at z, orient the other one pointing away from z.
        # It cannot be a collider because we have already oriented all unshielded triples that contain colliders.
        for z in potential_zs:
            z = graph.nodes[z]
            breakflag = False
            if graph.only_directed_edge_exists(x, z) and graph.undirected_edge_exists(
                z, y
            ):
                for node in graph.nodes:
                    if graph.only_directed_edge_exists(graph.nodes[node], y):
                        breakflag = True
                        break
                if breakflag is True:
                    continue
                return TestResult(
                    u=y,
                    v=z,
                    action=TestResultAction.REMOVE_EDGE_DIRECTED,
                    data={},
                )

            if graph.only_directed_edge_exists(y, z) and graph.undirected_edge_exists(
                z, x
            ):
                for node in graph.nodes:
                    if graph.only_directed_edge_exists(graph.nodes[node], x):
                        continue
                return TestResult(
                    u=x,
                    v=z,
                    action=TestResultAction.REMOVE_EDGE_DIRECTED,
                    data={},
                )


class FurtherOrientTripleTest(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    generator: Optional[GeneratorInterface] = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    chunk_size_parallel_processing: IntegerParameter = 1
    parallel: BoolParameter = False

    def process(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> Optional[List[TestResult] | TestResult]:
        """
        Further orientation rule.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """

        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        results = []
        for z in graph.nodes:
            z = graph.nodes[z]
            # check if it is a potential z
            if not (graph.edge_exists(y, z) and graph.edge_exists(x, z)):
                continue

            if (
                graph.undirected_edge_exists(x, y)
                and graph.only_directed_edge_exists(x, z)
                and graph.only_directed_edge_exists(z, y)
            ):
                results.append(
                    TestResult(
                        u=y,
                        v=x,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
            if (
                graph.undirected_edge_exists(x, y)
                and graph.only_directed_edge_exists(y, z)
                and graph.only_directed_edge_exists(z, x)
            ):
                results.append(
                    TestResult(
                        u=x,
                        v=y,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
        return results


class OrientQuadrupleTest(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    generator: Optional[GeneratorInterface] = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    chunk_size_parallel_processing: IntegerParameter = 1
    parallel: BoolParameter = False

    def process(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> Optional[List[TestResult] | TestResult]:
        """
        Further orientation rule.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """

        y = graph.nodes[nodes[0]]
        w = graph.nodes[nodes[1]]

        potential_zs = set()

        # TODO: just iterate over edges
        for z in graph.nodes:
            z = graph.nodes[z]
            if graph.edge_exists(y, z) and graph.edge_exists(z, w):
                potential_zs.add(z)

        results = []
        for zs in itertools.combinations(potential_zs, 2):
            x, z = zs
            if (
                not graph.edge_exists(y, w)
                and graph.only_directed_edge_exists(y, z)
                and graph.only_directed_edge_exists(w, z)
                and graph.undirected_edge_exists(x, y)
                and graph.undirected_edge_exists(x, w)
                and graph.undirected_edge_exists(x, z)
            ):
                results.append(
                    TestResult(
                        u=z,
                        v=x,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
            if (
                not graph.edge_exists(y, w)
                and graph.only_directed_edge_exists(y, x)
                and graph.only_directed_edge_exists(w, x)
                and graph.undirected_edge_exists(y, z)
                and graph.undirected_edge_exists(w, z)
                and graph.undirected_edge_exists(x, z)
            ):
                results.append(
                    TestResult(
                        u=x,
                        v=z,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
        return results


class FurtherOrientQuadrupleTest(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    generator: Optional[GeneratorInterface] = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    chunk_size_parallel_processing: int = 1
    parallel: bool = False

    def process(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> Optional[List[TestResult] | TestResult]:
        """
        Further orientation rule.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """

        x = graph.nodes[nodes[0]]
        w = graph.nodes[nodes[1]]

        potential_zs = set()

        # TODO: just iterate over edges
        for z in graph.nodes:
            z = graph.nodes[z]
            if graph.edge_exists(x, z) and graph.edge_exists(z, w):
                potential_zs.add(z)

        results = []
        for zs in itertools.combinations(potential_zs, 2):
            y, z = zs
            if (
                not graph.edge_exists(y, z)
                and graph.undirected_edge_exists(x, z)
                and graph.undirected_edge_exists(x, w)
                and graph.undirected_edge_exists(x, y)
                and graph.only_directed_edge_exists(y, w)
                and graph.only_directed_edge_exists(w, z)
            ):
                results.append(
                    TestResult(
                        u=z,
                        v=x,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
            elif (
                not graph.edge_exists(z, y)
                and graph.undirected_edge_exists(x, y)
                and graph.undirected_edge_exists(x, w)
                and graph.undirected_edge_exists(x, z)
                and graph.only_directed_edge_exists(z, w)
                and graph.only_directed_edge_exists(w, y)
            ):
                results.append(
                    TestResult(
                        u=y,
                        v=x,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
            elif (
                not graph.edge_exists(y, z)
                and graph.undirected_edge_exists(w, z)
                and graph.undirected_edge_exists(x, w)
                and graph.undirected_edge_exists(w, y)
                and graph.only_directed_edge_exists(y, x)
                and graph.only_directed_edge_exists(x, z)
            ):
                results.append(
                    TestResult(
                        u=z,
                        v=w,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
            elif (
                not graph.edge_exists(z, y)
                and graph.undirected_edge_exists(w, y)
                and graph.undirected_edge_exists(x, w)
                and graph.undirected_edge_exists(w, z)
                and graph.only_directed_edge_exists(z, x)
                and graph.only_directed_edge_exists(x, y)
            ):
                results.append(
                    TestResult(
                        u=y,
                        v=w,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
        return results
