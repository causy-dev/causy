import random

from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.graph_model import graph_model_factory
from causy.math_utils import sum_lists
from causy.independence_tests.common import (
    CorrelationCoefficientTest,
)
from causy.algorithms.fci import FCIEdgeType

from tests.utils import CausyTestCase


class IndependenceTestTestCase(CausyTestCase):
    def test_correlation_coefficient_standard_model(self):
        samples = {}
        test_data = []
        n = 1000
        x = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        noise_y = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        y = sum_lists([5 * x_val for x_val in x], noise_y)
        samples["u"] = x
        samples["v"] = y
        for i in range(n):
            entry = {}
            for key in samples.keys():
                entry[key] = samples[key][i]
            test_data.append(entry)

        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.1),
        ]
        model = graph_model_factory(pipeline_steps=pipeline)()
        model.create_graph_from_data(test_data)
        model.create_all_possible_edges()
        model.execute_pipeline_steps()
        self.assertEqual(len(model.graph.action_history[-1]["actions"]), 0)
