from typing import Tuple, List, Optional

from causy.generators import AllCombinationsGenerator
from causy.interfaces import (
    TestResultAction,
    IndependenceTestInterface,
    ComparisonSettings,
    BaseGraphInterface,
    TestResult,
)


class ColliderRuleFCI(IndependenceTestInterface):
    generator = AllCombinationsGenerator(
        comparison_settings=ComparisonSettings(min=2, max=2)
    )
    chunk_size_parallel_processing = 1
    parallel = False

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> Optional[List[TestResult] | TestResult]:
        """
        Some notes on how we implment FCI: After the independence tests, we have a graph with undirected edges which are
        implemented as two directed edges, one in each direction. We initialize the graph by adding values to all these edges,
        in the beginning, they get the value "either directed or undirected". Then we perform the collider test. Unlike in PC,
        we do not delete directed edges from z to x and from z to y in order to obtain the structure (x -> z <- y). Instead, we
        delete the information "either directed or undirected" from the directed edges from x to z and from y to z. That means,
        the directed edges from x to z and from y to z are now truly directed edges. The edges from z to x and from z to y can
        still stand for a directed edge or no directed edge. In the literature, this is portrayed by the meta symbol * and we
        obtain x *-> z <-* y. There might be ways to implement these similar but still subtly different orientation rules more consistently.

        TODO: write tests

        We call triples x, y, z of nodes v structures if x and y that are NOT adjacent but share an adjacent node z.
        V structures looks like this in the undirected skeleton: (x - z - y).
        We now check if z is in the separating set. If so, the edges must be oriented from x to z and from y to z:
        (x *-> z <-* y), where * indicated that there can be an arrowhead or none, we do not know, at least until
        applying further rules.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """

        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        # if x and y are adjacent, do nothing
        if graph.undirected_edge_exists(x, y):
            return TestResult(x=x, y=y, action=TestResultAction.DO_NOTHING, data={})

        # if x and y are NOT adjacent, store all shared adjacent nodes
        potential_zs = set(graph.edges[x.id].keys()).intersection(
            set(graph.edges[y.id].keys())
        )

        actions = graph.retrieve_edge_history(
            x, y, TestResultAction.REMOVE_EDGE_UNDIRECTED
        )

        # if x and y are not independent given z, safe action: make z a collider
        results = []
        for z in potential_zs:
            z = graph.nodes[z]

            separators = []
            for action in actions:
                if "separatedBy" in action.data:
                    separators += [a.id for a in action.data["separatedBy"]]

            if z.id not in separators:
                results += [
                    TestResult(
                        x=x,
                        y=z,
                        action=TestResultAction.UPDATE_EDGE_DIRECTED,
                        data={"edge_type": None},
                    ),
                    TestResult(
                        x=y,
                        y=z,
                        action=TestResultAction.UPDATE_EDGE_DIRECTED,
                        data={"edge_type": None},
                    ),
                ]
        return results
