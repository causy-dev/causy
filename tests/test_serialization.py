import unittest

from causy.cli import create_pipeline
from causy.graph import graph_model_factory
from causy.independence_tests import CalculateCorrelations, CorrelationCoefficientTest
from causy.serialization import serialize_model


class SerializationTestCase(unittest.TestCase):
    def test_serialize(self):
        pipeline = [CalculateCorrelations(), CorrelationCoefficientTest(threshold=0.1)]
        model = graph_model_factory(pipeline_steps=pipeline)()
        model_dict = serialize_model(model)
        self.assertEqual(len(model_dict["steps"]), 2)
        self.assertEqual(
            model_dict["steps"][0]["name"],
            "causy.independence_tests.CalculateCorrelations",
        )

    def test_serialize_and_load(self):
        print("test_serialize_and_load")
        pipeline = [CalculateCorrelations(), CorrelationCoefficientTest(threshold=0.1)]
        model = graph_model_factory(pipeline_steps=pipeline)()
        model_dict = serialize_model(model)
        pipeline = create_pipeline(model_dict)
        model = graph_model_factory(pipeline_steps=pipeline)()
        self.assertEqual(len(model.pipeline_steps), 2)
        self.assertEqual(
            model.pipeline_steps[0].__class__.__name__, "CalculateCorrelations"
        )
