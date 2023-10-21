from typing import Tuple, List
import itertools

from causy.generators import AllCombinationsGenerator
from causy.interfaces import (
    BaseGraphInterface,
    TestResult,
    TestResultAction,
    IndependenceTestInterface,
    ComparisonSettings,
)

# theory for all orientation rules with pictures: https://hpi.de/fileadmin/user_upload/fachgebiete/plattner/teaching/CausalInference/2019/Introduction_to_Constraint-Based_Causal_Structure_Learning.pdf


class ColliderTest(IndependenceTestInterface):
    GENERATOR = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    CHUNK_SIZE_PARALLEL_PROCESSING = 1
    PARALLEL = False

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> List[TestResult] | TestResult:
        """
        We call triples x, y, z of nodes v structures if x and y that are NOT adjacent but share an adjacent node z.
        V structures looks like this in the undirected skeleton: (x - z - y).
        We now check if z is in the separating set.
        If z is not in the separating set, we know that x and y are uncorrelated given z.
        So, the edges must be oriented from x to z and from y to z (x -> z <- y).
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """
        # https://github.com/pgmpy/pgmpy/blob/1fe10598df5430295a8fc5cdca85cf2d9e1c4330/pgmpy/estimators/PC.py#L416

        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        # if x and y are adjacent, do nothing
        if graph.undirected_edge_exists(x, y):
            return TestResult(x=x, y=y, action=TestResultAction.DO_NOTHING, data={})

        # if x and y are NOT adjacent, store all shared adjacent nodes
        potential_zs = set(graph.edges[x.id].keys()).intersection(
            set(graph.edges[y.id].keys())
        )

        # if x and y are not independent given z, safe action: make z a collider
        results = []
        for z in potential_zs:
            z = graph.nodes[z]
            actions = graph.retrieve_edge_history(
                x, y, TestResultAction.REMOVE_EDGE_UNDIRECTED
            )

            separators = []
            for action in actions:
                if "separatedBy" in action.data:
                    separators += [a.id for a in action.data["separatedBy"]]

            if z.id not in separators:
                results += [
                    TestResult(
                        x=z,
                        y=x,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    ),
                    TestResult(
                        x=z,
                        y=y,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    ),
                ]
        return results


class NonColliderTest(IndependenceTestInterface):
    GENERATOR = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    CHUNK_SIZE_PARALLEL_PROCESSING = 1
    PARALLEL = False

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> List[TestResult] | TestResult:
        """
        Further orientation rule: all v structures that are colliders are already oriented.
        We now orient all v structures that have a single alternative to being a collider.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """

        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        # if x and y are adjacent, do nothing
        if graph.edge_exists(x, y):
            return

        # if x and y are NOT adjacent, store all shared adjacent nodes
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
                    x=y,
                    y=z,
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
                    x=x,
                    y=z,
                    action=TestResultAction.REMOVE_EDGE_DIRECTED,
                    data={},
                )
        return


class FurtherOrientTripleTest(IndependenceTestInterface):
    GENERATOR = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    CHUNK_SIZE_PARALLEL_PROCESSING = 1
    PARALLEL = False

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> List[TestResult] | TestResult:
        """
        Further orientation rule.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """

        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        potential_zs = set(graph.edges[x.id].keys()).intersection(
            set(graph.edges[y.id].keys())
        )

        results = []
        for z in potential_zs:
            z = graph.nodes[z]
            if (
                graph.undirected_edge_exists(x, y)
                and graph.only_directed_edge_exists(x, z)
                and graph.only_directed_edge_exists(z, y)
            ):
                results.append(
                    TestResult(
                        x=y,
                        y=x,
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
                        x=x,
                        y=y,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
        return results


class OrientQuadrupleTest(IndependenceTestInterface):
    GENERATOR = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    CHUNK_SIZE_PARALLEL_PROCESSING = 1
    PARALLEL = False

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> List[TestResult] | TestResult:
        """
        Further orientation rule.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """

        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        potential_zs = set(graph.edges[x.id].keys()).intersection(
            set(graph.edges[y.id].keys())
        )

        results = []
        for zs in itertools.combinations(potential_zs, 2):
            z = graph.nodes[zs[0]]
            w = graph.nodes[zs[1]]
            if (
                not graph.undirected_edge_exists(x, y)
                and graph.only_directed_edge_exists(x, z)
                and graph.only_directed_edge_exists(y, z)
                and graph.undirected_edge_exists(x, w)
                and graph.undirected_edge_exists(y, w)
                and graph.undirected_edge_exists(z, w)
            ):
                results.append(
                    TestResult(
                        x=z,
                        y=w,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
            if (
                not graph.undirected_edge_exists(x, y)
                and graph.only_directed_edge_exists(x, w)
                and graph.only_directed_edge_exists(y, w)
                and graph.undirected_edge_exists(x, z)
                and graph.undirected_edge_exists(y, z)
                and graph.undirected_edge_exists(z, w)
            ):
                results.append(
                    TestResult(
                        x=w,
                        y=z,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
        return results


class FurtherOrientQuadrupleTest(IndependenceTestInterface):
    GENERATOR = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    CHUNK_SIZE_PARALLEL_PROCESSING = 1
    PARALLEL = False

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> List[TestResult] | TestResult:
        """
        Further orientation rule.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """

        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        potential_zs = set(graph.edges[x.id].keys()).intersection(
            set(graph.edges[y.id].keys())
        )

        results = []
        for zs in itertools.combinations(potential_zs, 2):
            z = graph.nodes[zs[0]]
            w = graph.nodes[zs[1]]
            if (
                not graph.undirected_edge_exists(x, y)
                and graph.only_directed_edge_exists(x, z)
                and graph.only_directed_edge_exists(z, y)
                and graph.undirected_edge_exists(z, w)
                and graph.undirected_edge_exists(x, z)
                and graph.undirected_edge_exists(y, z)
            ):
                results.append(
                    TestResult(
                        x=y,
                        y=z,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
            if (
                not graph.undirected_edge_exists(y, x)
                and graph.only_directed_edge_exists(y, z)
                and graph.only_directed_edge_exists(z, x)
                and graph.undirected_edge_exists(z, w)
                and graph.undirected_edge_exists(x, z)
                and graph.undirected_edge_exists(y, z)
            ):
                results.append(
                    TestResult(
                        x=x,
                        y=z,
                        action=TestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    )
                )
        return results
