import collections
import copy
import itertools
import logging
import random
from dataclasses import dataclass
from typing import Optional, Union, Dict

from pydantic import BaseModel

from causy.interfaces import (
    ComparisonSettings,
    GeneratorInterface,
    BaseGraphInterface,
    GraphModelInterface,
    AS_MANY_AS_FIELDS,
)
from causy.graph_utils import load_pipeline_artefact_by_definition

logger = logging.getLogger(__name__)


@dataclass
class AllCombinationsGenerator(GeneratorInterface):
    """
    Generates all combinations of nodes in the graph
    """

    def __init__(self, *args, **kwargs):
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

        # create all combinations
        for r in range(start, stop):
            # we need to sort the nodes to make sure we always get the same order of nodes - this is important for testing
            for i in itertools.combinations(graph.nodes, r):
                yield i


class PairsWithEdgesInBetweenGenerator(GeneratorInterface):
    chunk_size: int = 100
    chunked: Optional[bool] = None

    def __init__(
        self,
        *args,
        chunk_size: Optional[int] = None,
        chunked: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.chunked = chunked
        if chunk_size is not None:
            self.chunk_size = chunk_size

    def generate(
        self, graph: BaseGraphInterface, graph_model_instance_: GraphModelInterface
    ):
        local_edges = copy.deepcopy(graph.edges)

        edges = []

        for f_node in local_edges.keys():
            for t_node in graph.edges[f_node].keys():
                edges.append((f_node, t_node))

        if self.chunked:
            for i in range(0, len(edges), self.chunk_size):
                yield edges[i : i + self.chunk_size]

        for edge in edges:
            yield edge


class PairsWithNeighboursGenerator(GeneratorInterface):
    """
    Generates all combinations of pairs of nodes with their neighbours
    """

    shuffle_combinations: bool = True
    chunked: bool = True

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
        for i in range(start, stop):
            logger.debug(f"PairsWithNeighboursGenerator: i={i}")
            checked_combinations = set()
            local_edges = copy.deepcopy(dict(graph.edges))
            for node in local_edges.keys():
                local_edges[node] = copy.deepcopy(dict(local_edges[node]))
                for neighbour in local_edges[node].keys():
                    if (node, neighbour) in checked_combinations:
                        continue

                    checked_combinations.add((node, neighbour))
                    if i == 2:
                        yield (node, neighbour)
                        continue

                    other_neighbours = set(local_edges[node].keys())

                    if neighbour in other_neighbours:
                        other_neighbours.remove(neighbour)

                    combinations = list(itertools.combinations(other_neighbours, i - 2))
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

    every_nth: int = 100
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
