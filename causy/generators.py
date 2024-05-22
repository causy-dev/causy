import itertools
import logging
import random
from typing import Optional, Union, Dict

from pydantic import BaseModel

from causy.interfaces import (
    GeneratorInterface,
    BaseGraphInterface,
    GraphModelInterface,
    AS_MANY_AS_FIELDS,
)
from causy.models import ComparisonSettings
from causy.graph_utils import load_pipeline_artefact_by_definition
from causy.variables import IntegerParameter, BoolParameter

logger = logging.getLogger(__name__)


class AllCombinationsGenerator(GeneratorInterface):
    """
    Generates all combinations of nodes in the graph in accordance with the configured amount of nodes to compare.
    Yields node combinations as tuples, but only in one ordering, i.e. mathematically it returns sets.
    e.g. if your graph consists of the nodes [X, Y, Z] and you want to compare 2 nodes,
    it will yield (X, Y), (X, Z), (Y, Z). It will not additionally yield (Y, X), (Z, X), (Z, Y).
    """

    def generate(
        self, graph: BaseGraphInterface, graph_model_instance_: GraphModelInterface
    ):
        start = self.comparison_settings.min
        # if min (size of node sets to generate) is larger than the number of variables (represented by nodes) in our dataset, we can't create any combinations
        if start > len(graph.nodes):
            return

        # if max is AS_MANY_AS_FIELDS, we set it to number of variables + 1
        if self.comparison_settings.max == AS_MANY_AS_FIELDS:
            stop = len(graph.nodes) + 1
        else:
            stop = self.comparison_settings.max + 1

        # if start is higher than number of variables, we set it to the amount of variables
        if stop > len(graph.nodes) + 1:
            stop = len(graph.nodes) + 1

        # if stop is smaller than start, we can't create any combinations
        if stop < start:
            return

        # create all combinations
        for range_size in range(start, stop):
            for subset in itertools.combinations(graph.nodes, range_size):
                yield subset


class PairsWithEdgesInBetweenGenerator(GeneratorInterface):
    """
    Generates all pairs of nodes that have edges in between them. It does not matter if the edge is directed or not.
    However, if it is an edge which points in both/no directions, it will be iterated over them twice.
    """

    chunk_size: IntegerParameter = 100
    chunked: Optional[BoolParameter] = None

    def __init__(
        self,
        *args,
        chunk_size: Optional[IntegerParameter] = None,
        chunked: Optional[BoolParameter] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.chunked = chunked
        if chunk_size is not None:
            self.chunk_size = chunk_size

    def generate(
        self, graph: BaseGraphInterface, graph_model_instance_: GraphModelInterface
    ):
        edges = []

        for f_node in graph.edges.keys():
            for t_node in graph.edges[f_node].keys():
                if not graph.directed_edge_exists(f_node, t_node):
                    continue
                edges.append((f_node, t_node))

        if self.chunked:
            for i in range(0, len(edges), self.chunk_size):
                yield edges[i : i + self.chunk_size]

        for edge in edges:
            yield edge


class PairsWithNeighboursGenerator(GeneratorInterface):
    """
    Generates all combinations of pairs of nodes that are neighbours and the neighbours of the first input node.
    e.g. if your graph consists of the nodes [X, Y, Z, W, V] your output could be:
    [X, Y, neighbours(X)], [Y, X, neighbours(Y)], [X, Z, neighbours(X)], [Z, X, neighbours(Z)], ...
    (if, among others, X and Y are neighbours and X and Z are neighbours)
    """

    shuffle_combinations: BoolParameter = True
    chunked: BoolParameter = True

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def generate(
        self, graph: BaseGraphInterface, graph_model_instance_: GraphModelInterface
    ):
        start = self.comparison_settings.min
        # if min is longer then our dataset, we can't create any combinations
        if start > len(graph.nodes):
            return

        # if max is AS_MANY_AS_FIELDS, we set it to the length of the dataset + 1
        if self.comparison_settings.max == AS_MANY_AS_FIELDS:
            stop = len(graph.nodes) + 1
        else:
            stop = self.comparison_settings.max + 1

        # if start is longer then our dataset, we set it to the length of the dataset
        if stop > len(graph.nodes) + 1:
            stop = len(graph.nodes) + 1

        # if stop is smaller then start, we can't create any combinations
        if stop < start:
            return

        if start < 2:
            raise ValueError("PairsWithNeighboursGenerator: start must be at least 2")
        for range_size in range(start, stop):
            logger.info(f"range_size = {range_size}")
            logger.debug(f"PairsWithNeighboursGenerator: range_size={range_size}")
            checked_combinations = set()
            for node in graph.edges:
                for neighbour in graph.edges[node].keys():
                    if not graph.directed_edge_exists(node, neighbour):
                        continue

                    if (node, neighbour) in checked_combinations:
                        continue

                    checked_combinations.add((node, neighbour))
                    if range_size == 2:
                        yield [[node, neighbour]]
                        continue

                    other_neighbours = set(
                        [
                            k
                            for k, value in graph.edges[node].items()
                            if graph.directed_edge_exists(k, node)
                        ]
                    )
                    logger.info(f"other_neighbors before removal={other_neighbours}")

                    if neighbour in other_neighbours:
                        other_neighbours.remove(neighbour)
                    else:
                        logger.debug(
                            "PairsWithNeighboursGenerator: neighbour not in other_neighbours. This should not happen."
                        )
                    logger.info(f"node={node}, neighbour={neighbour}")
                    logger.info(f"other_neighbours={other_neighbours}")
                    combinations = list(
                        itertools.combinations(other_neighbours, range_size - 2)
                    )

                    if self.shuffle_combinations:
                        combinations = list(combinations)
                        import random

                        random.shuffle(combinations)

                    if self.chunked:
                        chunk = []
                        for k in combinations:
                            chunk.append([node, neighbour] + [ks for ks in k])
                        yield chunk
                    else:
                        for k in combinations:
                            yield [node, neighbour] + [ks for ks in k]


class RandomSampleGenerator(GeneratorInterface, BaseModel):
    """
    Executes another generator and returns a random sample of the results
    """

    every_nth: IntegerParameter = 100
    generator: Optional[GeneratorInterface] = None

    def __init__(
        self,
        *args,
        generator: Optional[Union[GeneratorInterface, Dict[any, any]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if generator is not None:
            if isinstance(generator, GeneratorInterface):
                self.generator = generator
            else:
                self.generator = load_pipeline_artefact_by_definition(generator)
        else:
            raise ValueError("RandomSampleGenerator: generator must be set")

    def generate(self, graph: BaseGraphInterface, graph_model_instance_: dict):
        """
        Executes another generator and returns a random sample of the results
        :param graph:
        :param graph_model_instance_:
        :return: yields a random sample of the results
        """
        for combination in self.generator.generate(graph, graph_model_instance_):
            if random.randint(0, self.every_nth) == 0:
                yield combination
