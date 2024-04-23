from causy.cli import create_pipeline
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.graph_model import graph_model_factory
from causy.independence_tests.common import CorrelationCoefficientTest
from causy.serialization import serialize_model

from tests.utils import CausyTestCase


class SerializationTestCase(CausyTestCase):
    def test_serialize(self):
        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.1),
        ]
        model = graph_model_factory(pipeline_steps=pipeline)()
        model_dict = serialize_model(model)
        self.assertEqual(len(model_dict["steps"]), 2)
        self.assertEqual(
            model_dict["steps"][0]["name"],
            "causy.common_pipeline_steps.calculation.CalculatePearsonCorrelations",
        )

    def test_serialize_and_load(self):
        pipeline = [
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.1),
        ]
        model = graph_model_factory(pipeline_steps=pipeline)()
        model_dict = serialize_model(model)
        pipeline = create_pipeline(model_dict)
        model = graph_model_factory(pipeline_steps=pipeline)()
        self.assertEqual(len(model.pipeline_steps), 2)
        self.assertEqual(
            model.pipeline_steps[0].__class__.__name__, "CalculatePearsonCorrelations"
        )
