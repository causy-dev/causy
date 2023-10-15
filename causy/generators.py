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
