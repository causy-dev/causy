import unittest
import torch

from causy.sample_generator import (
    TimeseriesSampleGenerator,
    random,
    SampleLaggedEdge,
    IIDSampleGenerator,
    NodeReference,
)

from causy.sample_generator import SampleEdge


class TimeSeriesSampleGeneratorTest(unittest.TestCase):
    def test_iid_sample_generator_without_randomness(self):
        self.assertTrue(True)
        # TODO: fix bug in iid sample generator and write test

    def test_timeseries_sample_generator(self):
        model_one = TimeseriesSampleGenerator(
            edges=[
                SampleLaggedEdge(NodeReference("X", -1), NodeReference("X"), 0.9),
                SampleLaggedEdge(NodeReference("Y", -1), NodeReference("Y"), 0.9),
                SampleLaggedEdge(NodeReference("Z", -1), NodeReference("Z"), 0.9),
                SampleLaggedEdge(NodeReference("Z", -1), NodeReference("Y"), 5),
                SampleLaggedEdge(NodeReference("Y", -1), NodeReference("X"), 7),
            ],
            random=lambda: 0,
        )

        model_one._initial_distribution_fn = lambda x: torch.tensor(
            1, dtype=torch.float32
        )

        result, graph = model_one.generate(100)
        result_10_samples, _ = model_one.generate(10)
        self.assertEqual(len(result["X"]), 100)
        list_of_ground_truth_values = [
            1.0000e00,
            7.9000e00,
            4.8410e01,
            1.1224e02,
            1.9117e02,
            2.7870e02,
            3.6978e02,
            4.6053e02,
            5.4803e02,
            6.3016e02,
        ]
        for i in range(10):
            self.assertAlmostEqual(
                result_10_samples["X"].tolist()[i],
                list_of_ground_truth_values[i],
                delta=0.005,
            )

    def test_without_randomness(self):
        model = TimeseriesSampleGenerator(
            edges=[
                SampleLaggedEdge(NodeReference("X", -1), NodeReference("X"), 0.9),
                SampleLaggedEdge(NodeReference("Y", -1), NodeReference("Y"), 0.9),
                SampleLaggedEdge(NodeReference("X", -1), NodeReference("Y"), 5),
            ],
            random=lambda: 0,
        )
        model._initial_distribution_fn = lambda x: torch.tensor(1, dtype=torch.float32)

        result, graph = model.generate(100)

        self.assertEqual(len(result["X"]), 100)
        self.assertEqual(len(result["Y"]), 100)
        self.assertEqual(result["X"][0], 1)
        self.assertEqual(result["Y"][0], 1)
        self.assertAlmostEqual(result["X"][1].item(), 0.9, places=2)
        self.assertAlmostEqual(result["Y"][1].item(), 5.9, places=2)
        self.assertAlmostEqual(result["X"][2].item(), 0.81, places=2)
        self.assertAlmostEqual(result["Y"][2].item(), 9.81, places=2)

    def test_generating_initial_values(self):
        model = TimeseriesSampleGenerator(
            edges=[
                SampleLaggedEdge(NodeReference("X", -1), NodeReference("X"), 0.9),
                SampleLaggedEdge(NodeReference("Y", -1), NodeReference("Y"), 0.9),
                SampleLaggedEdge(NodeReference("X", -1), NodeReference("Y"), 5),
            ],
        )
        model._initial_distribution_fn = lambda x: x
        initial_values = model._calculate_initial_values()

        self.assertAlmostEqual(float(initial_values["X"] ** 2), 5.2630, places=0)
        self.assertAlmostEqual(float(initial_values["Y"] ** 2), 6602.2842, places=0)

    def test_multiple_autocorrelations(self):
        model_multi_autocorr = TimeseriesSampleGenerator(
            edges=[
                SampleLaggedEdge(NodeReference("X", -1), NodeReference("X"), 0.4),
                SampleLaggedEdge(NodeReference("X", -2), NodeReference("X"), 0.4),
            ],
            random=lambda: torch.tensor(0, dtype=torch.float32),
        )

        model_multi_autocorr._initial_distribution_fn = lambda x: torch.tensor(
            1, dtype=torch.float32
        )

        result = model_multi_autocorr._generate_data(100)
        self.assertEqual(len(result["X"]), 100)
        self.assertAlmostEqual(result["X"][0].item(), 1, places=2)
        self.assertAlmostEqual(result["X"][1].item(), 0.8, places=2)
        self.assertAlmostEqual(result["X"][2].item(), 0.72, places=2)
        self.assertAlmostEqual(result["X"][3].item(), 0.608, places=2)
