from typing import Tuple, List, Optional, Generic

from causy.generators import AllCombinationsGenerator
from causy.interfaces import (
    PipelineStepInterface,
    BaseGraphInterface,
    GeneratorInterface,
    PipelineStepInterfaceType,
)
from causy.models import ComparisonSettings, TestResultAction, TestResult


class ColliderRuleFCI(
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
        Some notes on how we implment FCI: After the independence tests, we have a graph with undirected edges which are
        implemented as two directed edges, one in each direction. We initialize the graph by adding values to all these edges,
        in the beginning, they get the value "either directed or undirected". Then we perform the collider test. Unlike in PC,
        we do not delete directed edges from z to u and from z to v in order to obtain the structure (u -> z <- v). Instead, we
        delete the information "either directed or undirected" from the directed edges from u to z and from v to z. That means,
        the directed edges from u to z and from v to z are now truly directed edges. The edges from z to u and from z to v can
        still stand for a directed edge or no directed edge. In the literature, this is portrayed by the meta symbol * and we
        obtain u *-> z <-* v. There might be ways to implement these similar but still subtly different orientation rules more consistently.

        TODO: write tests

        We call triples u, v, z of nodes v structures if u and v that are NOT adjacent but share an adjacent node z.
        V structures looks like this in the undirected skeleton: (u - z - v).
        We now check if z is in the separating set. If so, the edges must be oriented from u to z and from v to z:
        (u *-> z <-* v), where * indicated that there can be an arrowhead or none, we do not know, at least until
        applying further rules.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """

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
                results += [
                    TestResult(
                        u=x,
                        v=z,
                        action=TestResultAction.UPDATE_EDGE_DIRECTED,
                        data={"edge_type": None},
                    ),
                    TestResult(
                        u=y,
                        v=z,
                        action=TestResultAction.UPDATE_EDGE_DIRECTED,
                        data={"edge_type": None},
                    ),
                ]
        return results
