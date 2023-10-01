from typing import Tuple, List
import itertools

from interfaces import BaseGraphInterface, CorrelationTestResult, CorrelationTestResultAction, IndependenceTestInterface


class UnshieldedTripleColliderTest(IndependenceTestInterface):
    NUM_OF_COMPARISON_ELEMENTS = 2
    CHUNK_SIZE_PARALLEL_PROCESSING = 1

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> List[CorrelationTestResult] | CorrelationTestResult:
        """
        For all nodes x and y that are not adjacent but share an adjacent node z, we check if z is in the seperating set. 
        If z is not in the seperating set, we know that x and y are uncorrelated given z, so the edges must be oriented from x to z and from y to z.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """
        # https://github.com/pgmpy/pgmpy/blob/1fe10598df5430295a8fc5cdca85cf2d9e1c4330/pgmpy/estimators/PC.py#L416

        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        # if x and y are adjacent, do nothing
        if graph.edge_exists(x, y):
            return CorrelationTestResult(
                x=x, y=y, action=CorrelationTestResultAction.DO_NOTHING, data={}
            )

        # if x and y are NOT adjacent, store all shared adjacent nodes
        potential_zs = set(graph.edges[x].keys()).intersection(
            set(graph.edges[y].keys())
        )

        # if x and y are not independent given z, safe action: make z a collider
        results = []
        for z in potential_zs:
            separators = graph.retrieve_edge_history(
                x, y, CorrelationTestResultAction.REMOVE_EDGE_UNDIRECTED
            )

            if z not in separators:
                results += [
                    CorrelationTestResult(
                        x=z,
                        y=x,
                        action=CorrelationTestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    ),
                    CorrelationTestResult(
                        x=z,
                        y=y,
                        action=CorrelationTestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    ),
                ]
        return results



class UnshieldedTripleNonColliderTest(IndependenceTestInterface):
    NUM_OF_COMPARISON_ELEMENTS = 2
    CHUNK_SIZE_PARALLEL_PROCESSING = 1

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> List[CorrelationTestResult] | CorrelationTestResult:
        """
        Further orientation rule.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """

        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        # if x and y are adjacent, do nothing
        if graph.edge_exists(x, y):
            return CorrelationTestResult(
                x=x, y=y, action=CorrelationTestResultAction.DO_NOTHING, data={}
            )

        # if x and y are NOT adjacent, store all shared adjacent nodes
        potential_zs = set(graph.edges[x].keys()).intersection(
            set(graph.edges[y].keys())
        )
        results = []
        for z in potential_zs:
            if graph.directed_edge_exists(x, z):
                if graph.edge_exists(y, z):
                    results.append(CorrelationTestResult(
                        x=z,
                        y=y,
                        action=CorrelationTestResultAction.REMOVE_EDGE_DIRECTED,
                        data={},
                    ))
        return results



class FurtherOrientTripleTest(IndependenceTestInterface):
    NUM_OF_COMPARISON_ELEMENTS = 2
    CHUNK_SIZE_PARALLEL_PROCESSING = 1

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> List[CorrelationTestResult] | CorrelationTestResult:
        """
        Further orientation rule.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """
        
        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        potential_zs = set(graph.edges[x].keys()).intersection(
            set(graph.edges[y].keys())
        )

        results = []
        for z in potential_zs:
            if graph.edge_exists(x,y) and graph.directed_edge_exists(x,z) and graph.directed_edge_exists(z,y):
                results.append(CorrelationTestResult(
                            x=y,
                            y=x,
                            action=CorrelationTestResultAction.REMOVE_EDGE_DIRECTED,
                            data={},
                        ))
            if graph.edge_exists(x,y) and graph.directed_edge_exists(y,z) and graph.directed_edge_exists(z,x):
                results.append(CorrelationTestResult(
                            x=x,
                            y=y,
                            action=CorrelationTestResultAction.REMOVE_EDGE_DIRECTED,
                            data={},
                        ))
        return results
    

class OrientQuadrupleTest(IndependenceTestInterface):
    NUM_OF_COMPARISON_ELEMENTS = 2
    CHUNK_SIZE_PARALLEL_PROCESSING = 1

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> List[CorrelationTestResult] | CorrelationTestResult:
        """
        Further orientation rule.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """
        
        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        potential_zs = set(graph.edges[x].keys()).intersection(
            set(graph.edges[y].keys())
        )

        results = []
        for z in potential_zs:
            if graph.edge_exists(x,y) and graph.directed_edge_exists(x,z) and graph.directed_edge_exists(z,y):
                results.append(CorrelationTestResult(
                            x=y,
                            y=x,
                            action=CorrelationTestResultAction.REMOVE_EDGE_DIRECTED,
                            data={},
                        ))
            if graph.edge_exists(x,y) and graph.directed_edge_exists(y,z) and graph.directed_edge_exists(z,x):
                results.append(CorrelationTestResult(
                            x=x,
                            y=y,
                            action=CorrelationTestResultAction.REMOVE_EDGE_DIRECTED,
                            data={},
                        ))
        return results
    
class OrientQuadrupleTest(IndependenceTestInterface):
    NUM_OF_COMPARISON_ELEMENTS = 2
    CHUNK_SIZE_PARALLEL_PROCESSING = 1

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> List[CorrelationTestResult] | CorrelationTestResult:
        """
        Further orientation rule.
        :param nodes: list of nodes
        :param graph: the current graph
        :returns: list of actions that will be executed on graph
        """
        
        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]

        potential_zs = set(graph.edges[x].keys()).intersection(
            set(graph.edges[y].keys())
        )

        results = []
        for zs in itertools.combinations(potential_zs, 2):
            z = zs[0]
            w = zs[1]
            if not graph.edge_exists(x,y) and graph.directed_edge_exists(x,z) and graph.directed_edge_exists(y,z) \
                and graph.edge_exists(x,w) and graph.edge_exists(y,w):
                results.append(CorrelationTestResult(
                            x=z,
                            y=w,
                            action=CorrelationTestResultAction.REMOVE_EDGE_DIRECTED,
                            data={},
                        ))
            if not graph.edge_exists(x,y) and graph.directed_edge_exists(x,w) and graph.directed_edge_exists(y,w) \
                and graph.edge_exists(x,z) and graph.edge_exists(y,z):
                results.append(CorrelationTestResult(
                            x=w,
                            y=z,
                            action=CorrelationTestResultAction.REMOVE_EDGE_DIRECTED,
                            data={},
                        ))
        return results