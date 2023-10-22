import unittest
from statistics import correlation

from causy.independence_tests import CalculateCorrelations
from causy.utils import (
    pearson_correlation,
    serialize_module_name,
    load_pipeline_artefact_by_definition,
    load_pipeline_steps_by_definition,
)
import torch


class UtilsTestCase(unittest.TestCase):
    def test_pearson_correlation(self):
        self.assertEqual(
            pearson_correlation(
                torch.tensor([1, 2, 3], dtype=torch.float64),
                torch.tensor([1, 2, 3], dtype=torch.float64),
            ).item(),
            1,
        )
        self.assertEqual(
            pearson_correlation(
                torch.tensor([1, 2, 3], dtype=torch.float64),
                torch.tensor([3, 2, 1], dtype=torch.float64),
            ).item(),
            -1,
        )

    def test_serialize_module_name(self):
        self.assertEqual(serialize_module_name(self), "tests.test_utils.UtilsTestCase")

    def test_load_pipeline_artefact_by_definition(self):
        step = {"name": "causy.independence_tests.CalculateCorrelations"}
        self.assertIsInstance(
            load_pipeline_artefact_by_definition(step), CalculateCorrelations
        )

    def load_pipeline_steps_by_definition(self):
        steps = [{"name": "causy.independence_tests.CalculateCorrelations"}]
        self.assertIsInstance(
            load_pipeline_steps_by_definition(steps)[0], CalculateCorrelations
        )


if __name__ == "__main__":
    unittest.main()
