import itertools

from causy.interfaces import (
    ComparisonSettings,
    GeneratorInterface,
    BaseGraphInterface,
    GraphModelInterface,
    AS_MANY_AS_FIELDS,
)


class AllCombinationsGenerator(GeneratorInterface):
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

    def __init__(self, comparison_settings: ComparisonSettings):
        self.comparison_settings = comparison_settings


class PairsWithNeighboursGenerator(GeneratorInterface):
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
            checked_combinations = set()
            for node in graph.edges:
                for neighbour in graph.edges[node]:
                    if (neighbour, node) in checked_combinations or (
                        node,
                        neighbour,
                    ) in checked_combinations:
                        continue

                    checked_combinations.add((node, neighbour))
                    if i == 2:
                        yield (node.name, neighbour.name)
                        continue

                    other_neighbours = set(graph.edges[node]) | set(
                        graph.edges[neighbour]
                    )
                    other_neighbours.remove(node)
                    other_neighbours.remove(neighbour)

                    if len(other_neighbours) <= i:
                        continue

                    for k in itertools.combinations(other_neighbours, i):
                        yield [node.name, neighbour.name] + [ks.name for ks in k]

    def __init__(self, comparison_settings: ComparisonSettings):
        self.comparison_settings = comparison_settings
