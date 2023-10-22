import copy
import itertools
import logging
import random

from causy.interfaces import (
    ComparisonSettings,
    GeneratorInterface,
    BaseGraphInterface,
    GraphModelInterface,
    AS_MANY_AS_FIELDS,
)
from causy.utils import load_pipeline_artefact_by_definition

logger = logging.getLogger(__name__)


class AllCombinationsGenerator(GeneratorInterface):
    """
    Generates all combinations of nodes in the graph
    """

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
            for i in itertools.combinations(graph.nodes, r):
                yield i


class PairsWithNeighboursGenerator(GeneratorInterface):
    """
    Generates all combinations of pairs of nodes with their neighbours
    """

    shuffle_combinations = True
    chunked = True

    def __init__(
        self,
        comparison_settings: ComparisonSettings,
        chunked: bool = None,
        shuffle_combinations: bool = None,
    ):
        super().__init__(comparison_settings, chunked)
        if shuffle_combinations is not None:
            self.shuffle_combinations = shuffle_combinations

        if chunked is not None:
            self.chunked = chunked

    def serialize(self):
        result = super().serialize()
        result["params"]["shuffle_combinations"] = self.shuffle_combinations
        return result

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
            local_edges = copy.deepcopy(graph.edges)
            for node in local_edges:
                for neighbour in local_edges[node]:
                    if (node, neighbour) in checked_combinations:
                        continue

                    checked_combinations.add((node, neighbour))
                    if i == 2:
                        yield (node, neighbour)
                        continue

                    other_neighbours = set(graph.edges[node])
                    if neighbour in other_neighbours:
                        other_neighbours.remove(neighbour)
                    else:
                        continue
                    if len(other_neighbours) + 2 < i:
                        continue
                    combinations = itertools.combinations(other_neighbours, i)
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


class RandomSampleGenerator(GeneratorInterface):
    """
    Executes another generator and returns a random sample of the results
    """

    every_nth = 100

    def __init__(
        self,
        comparison_settings: ComparisonSettings = None,
        chunked: bool = None,
        every_nth: int = None,
        generator: GeneratorInterface = None,
    ):
        super().__init__(comparison_settings, chunked)
        if every_nth is not None:
            self.every_nth = every_nth

        if generator is not None:
            if isinstance(generator, GeneratorInterface):
                self.generator = generator
            else:
                self.generator = load_pipeline_artefact_by_definition(generator)
        else:
            raise ValueError("RandomSampleGenerator: generator must be set")

    def serialize(self):
        result = super().serialize()
        result["params"]["every_nth"] = self.every_nth
        result["params"]["generator"] = self.generator.serialize()
        return result

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
