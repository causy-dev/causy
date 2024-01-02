import unittest

from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.graph_utils import (
    serialize_module_name,
    load_pipeline_artefact_by_definition,
    load_pipeline_steps_by_definition,
)


class UtilsTestCase(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
