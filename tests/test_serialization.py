from causy.algorithms.pc import PC_DEFAULT_THRESHOLD
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.graph_model import graph_model_factory
from causy.independence_tests.common import CorrelationCoefficientTest
from causy.models import CausyAlgorithm
from causy.serialization import serialize_algorithm, load_algorithm_from_specification

from tests.utils import CausyTestCase


class SerializationTestCase(CausyTestCase):
    def test_serialize(self):
        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=PC_DEFAULT_THRESHOLD),
        ]
        model = graph_model_factory(
            CausyAlgorithm(
                pipeline_steps=pipeline,
                edge_types=[],
                name="test_serialize",
            )
        )()
        model_dict = serialize_algorithm(model)
        self.assertEqual(len(model_dict["pipeline_steps"]), 2)
        self.assertEqual(
            model_dict["pipeline_steps"][0]["name"],
            "causy.common_pipeline_steps.calculation.CalculatePearsonCorrelations",
        )

    def test_serialize_and_load(self):
        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=PC_DEFAULT_THRESHOLD),
        ]
        model = graph_model_factory(
            CausyAlgorithm(
                pipeline_steps=pipeline,
                edge_types=[],
                name="test_serialize",
            )
        )()
        algo_dict = serialize_algorithm(model)
        algorithm = load_algorithm_from_specification(algo_dict)
        model = graph_model_factory(algorithm=algorithm)()
        self.assertEqual(len(model.pipeline_steps), 2)
        self.assertEqual(
            model.pipeline_steps[0].__class__.__name__, "CalculatePearsonCorrelations"
        )
