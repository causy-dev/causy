import unittest
import torch

from causy.sample_generator import (
    TimeseriesSampleGenerator,
    random,
    SampleLaggedEdge,
    IIDSampleGenerator,
)

from causy.sample_generator import SampleEdge


class TimeSeriesSampleGeneratorTest(unittest.TestCase):
    def test_iid_sample_generator_without_randomness(self):
        self.assertTrue(True)
        # TODO: fix bug in iid sample generator and write test

    def test_timeseries_sample_generator(self):
        MODEL_ONE = TimeseriesSampleGenerator(
            initial_values={
                "Z": 1,
                "Y": 2,
                "X": 3,
            },
            variables={
                "alpha": 0.9,
                "beta": 0.9,
                "gamma": 0.9,
                "param_1": 5,
                "param_2": 7,
            },
            # generate the dependencies of variables on past values of themselves and other variables
            generators={
                "Z": lambda t, i: i.Z.t(-1) * i.alpha,
                "Y": lambda t, i: i.Y.t(-1) * i.beta + i.param_1 * i.Z.t(-1),
                "X": lambda t, i: i.X.t(-1) * i.gamma + i.param_2 * i.Y.t(-1),
            },
            edges=[
                SampleLaggedEdge("X", "X", 1),
                SampleLaggedEdge("Y", "Y", 1),
                SampleLaggedEdge("Z", "Z", 1),
                SampleLaggedEdge("Z", "Y", 1),
                SampleLaggedEdge("Y", "X", 1),
            ],
        )

        result, graph = MODEL_ONE.generate(100)
        result_10_samples, _ = MODEL_ONE.generate(10)
        self.assertEqual(len(result["X"]), 100)
        list_of_ground_truth_values = [
            3.0000,
            16.7000,
            62.6300,
            130.7070,
            212.8923,
            302.8484,
            395.6479,
            487.5262,
            575.6727,
            658.0551,
        ]
        for i in range(10):
            self.assertAlmostEqual(
                result_10_samples["X"].tolist()[i],
                list_of_ground_truth_values[i],
                delta=0.0005,
            )

    def test_without_randomness(self):
        MODEL = TimeseriesSampleGenerator(
            initial_values={
                "X": 1,
                "Y": 1,
            },
            variables={
                "alpha": 0.9,
                "beta": 0.9,
                "param_1": 5,
            },
            # generate the dependencies of variables on past values of themselves and other variables
            generators={
                "X": lambda t, i: i.X.t(-1) * i.alpha,
                "Y": lambda t, i: i.Y.t(-1) * i.beta + i.param_1 * i.X.t(-1),
            },
            edges=[
                SampleLaggedEdge("X", "X", 1),
                SampleLaggedEdge("Y", "Y", 1),
                SampleLaggedEdge("X", "Y", 1),
            ],
        )
        result, graph = MODEL.generate(100)

        self.assertEqual(len(result["X"]), 100)
        self.assertEqual(len(result["Y"]), 100)
        self.assertEqual(result["X"][0], 1)
        self.assertEqual(result["Y"][0], 1)
        self.assertAlmostEqual(result["X"][1].item(), 0.9, places=2)
        self.assertAlmostEqual(result["Y"][1].item(), 5.9, places=2)
        self.assertAlmostEqual(result["X"][2].item(), 0.81, places=2)
        self.assertAlmostEqual(result["Y"][2].item(), 9.81, places=2)
