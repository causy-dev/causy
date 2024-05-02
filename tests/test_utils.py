import copy
from unittest import skip

from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.graph_utils import (
    serialize_module_name,
    load_pipeline_artefact_by_definition,
    load_pipeline_steps_by_definition,
)
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference

from tests.utils import CausyTestCase


class UtilsTestCase(CausyTestCase):
    def test_serialize_module_name(self):
        self.assertEqual(serialize_module_name(self), "tests.test_utils.UtilsTestCase")

    def test_load_pipeline_artefact_by_definition(self):
        step = {
            "name": "causy.common_pipeline_steps.calculation.CalculatePearsonCorrelations"
        }
        self.assertIsInstance(
            load_pipeline_artefact_by_definition(step), CalculatePearsonCorrelations
        )

    def load_pipeline_steps_by_definition(self):
        steps = [
            {
                "name": "causy.common_pipeline_steps.calculation.CalculatePearsonCorrelations"
            }
        ]
        self.assertIsInstance(
            load_pipeline_steps_by_definition(steps)[0], CalculatePearsonCorrelations
        )

    def test_tests(self):
        model_one = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("Z"), NodeReference("X"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("V"), NodeReference("Z"), 1),
            ]
        )

        c, d, e, f, g = 2, 3, 4, 5, 6
        model_two = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("A"), NodeReference("C"), 1),
                SampleEdge(NodeReference("B"), NodeReference("C"), c),
                SampleEdge(NodeReference("A"), NodeReference("D"), d),
                SampleEdge(NodeReference("B"), NodeReference("D"), 1),
                SampleEdge(NodeReference("C"), NodeReference("D"), 1),
                SampleEdge(NodeReference("B"), NodeReference("E"), e),
                SampleEdge(NodeReference("E"), NodeReference("F"), f),
                SampleEdge(NodeReference("B"), NodeReference("F"), g),
                SampleEdge(NodeReference("C"), NodeReference("F"), 1),
                SampleEdge(NodeReference("D"), NodeReference("F"), 1),
            ],
        )

        _, g1 = model_one.generate(10000)
        g1 = copy.deepcopy(g1)
        _, g2 = model_two.generate(10000)

        with self.assertRaises(AssertionError):
            self.assertGraphStructureIsEqual(g1, g2)

        self.assertGraphStructureIsEqual(g1, g1)
