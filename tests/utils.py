import random
from unittest import TestCase
from unittest.util import safe_repr

import numpy as np
import torch


class CausyTestCase(TestCase):
    SEED = 42

    def setUp(self):
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.SEED)
        self.seeded_random = random.Random(self.SEED)

    def assertGraphStructureIsEqual(self, graph1, graph2, msg=None):
        for node_from in graph1.edges:
            for node_to in graph1.edges[node_from]:
                if (
                    node_from not in graph2.edges
                    or node_to not in graph2.edges[node_from].keys()
                ) and (
                    node_to not in graph2.edges
                    or node_from not in graph2.edges[node_to].keys()
                ):
                    msg = self._formatMessage(
                        msg,
                        f"{safe_repr(graph1)} is not equal to the structure of {safe_repr(graph1)}. Edge {node_from} - {node_to} is missing in {safe_repr(graph2)} (graph2).",
                    )
                    raise self.failureException(msg)

        for node_from in graph2.edges:
            for node_to in graph2.edges[node_from]:
                if (
                    node_from not in graph1.edges
                    or node_to not in graph1.edges[node_from].keys()
                ) and (
                    node_to not in graph1.edges
                    or node_from not in graph1._reverse_edges[node_to].keys()
                ):
                    msg = self._formatMessage(
                        msg,
                        f"{safe_repr(graph1)} is not equal to the structure of {safe_repr(graph1)}. Edge {node_from} - {node_to} is missing in {safe_repr(graph1)} (graph1).",
                    )
                    raise self.failureException(msg)

    def assertGraphStructureIsIn(self, graph1, graph2, msg=None):
        for node_from in graph2.edges:
            for node_to in graph2.edges[node_from]:
                if (
                    node_from not in graph1.edges
                    or node_to not in graph1.edges[node_from].keys()
                ):
                    msg = self._formatMessage(
                        msg,
                        f"{safe_repr(graph1)} is not equal to the structure of {safe_repr(graph1)}. Edge {node_from} -> {node_to} is missing in {safe_repr(graph1)} (graph1).",
                    )
                    raise self.failureException(msg)

    def tearDown(self):
        pass
