import torch

from causy.sample_generator import (
    TimeseriesSampleGenerator,
    random_normal,
    SampleEdge,
    IIDSampleGenerator,
    TimeAwareNodeReference,
    NodeReference,
)

from tests.utils import CausyTestCase


class TimeSeriesSampleGeneratorTest(CausyTestCase):
    def test_iid_sample_generator(self):
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
            ]
        )
        result = model._generate_data(100)
        self.assertEqual(list(result["X"].shape), [100])

    def test_iid_sample_generator_without_randomness(self):
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
            ]
        )
        model.random_fn = lambda: torch.tensor(1, dtype=torch.float32)

        result = model._generate_data(100)
        self.assertEqual(list(result["X"].shape), [100])
        self.assertEqual(result["X"][0], 1)
        self.assertEqual(result["X"][20], 1)
        self.assertEqual(result["Y"][0], 6)
        self.assertEqual(result["Y"][20], 6)
        self.assertEqual(result["Z"][0], 43)
        self.assertEqual(result["Z"][20], 43)

    def test_iid_sample_generator_more_complex_case_without_randomness(self):
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("W"), NodeReference("Y"), 5),
            ]
        )
        model.random_fn = lambda: torch.tensor(1, dtype=torch.float32)

        result = model._generate_data(100)
        self.assertEqual(list(result["X"].shape), [100])
        self.assertEqual(result["X"][0], 1)
        self.assertEqual(result["X"][20], 1)
        self.assertEqual(result["W"][0], 1)
        self.assertEqual(result["W"][20], 1)
        self.assertEqual(result["Y"][0], 11)
        self.assertEqual(result["Y"][20], 11)
        self.assertEqual(result["Z"][0], 12)
        self.assertEqual(result["Z"][20], 12)

    def test_timeseries_sample_generator_fixed_initial_distribution(self):
        model_one = TimeseriesSampleGenerator(
            edges=[
                SampleEdge(
                    TimeAwareNodeReference("X", -1), TimeAwareNodeReference("X"), 0.9
                ),
                SampleEdge(
                    TimeAwareNodeReference("Y", -1), TimeAwareNodeReference("Y"), 0.9
                ),
                SampleEdge(
                    TimeAwareNodeReference("Z", -1), TimeAwareNodeReference("Z"), 0.9
                ),
                SampleEdge(
                    TimeAwareNodeReference("Z", -1), TimeAwareNodeReference("Y"), 5
                ),
                SampleEdge(
                    TimeAwareNodeReference("Y", -1), TimeAwareNodeReference("X"), 7
                ),
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
                SampleEdge(
                    TimeAwareNodeReference("X", -1), TimeAwareNodeReference("X"), 0.9
                ),
                SampleEdge(
                    TimeAwareNodeReference("Y", -1), TimeAwareNodeReference("Y"), 0.9
                ),
                SampleEdge(
                    TimeAwareNodeReference("X", -1), TimeAwareNodeReference("Y"), 5
                ),
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

    def test_data_generator_multiple_autocorrelations(self):
        model_multi_autocorr = TimeseriesSampleGenerator(
            edges=[
                SampleEdge(
                    TimeAwareNodeReference("X", -1), TimeAwareNodeReference("X"), 0.4
                ),
                SampleEdge(
                    TimeAwareNodeReference("X", -2), TimeAwareNodeReference("X"), 0.4
                ),
            ],
            random=lambda: torch.tensor(0, dtype=torch.float32),
        )

        model_multi_autocorr._initial_distribution_fn = lambda x: torch.tensor(
            1, dtype=torch.float32
        )

        result, graph = model_multi_autocorr.generate(100)
        self.assertEqual(len(result["X"]), 100)
        self.assertAlmostEqual(result["X"][0].item(), 1, places=2)
        self.assertAlmostEqual(result["X"][1].item(), 0.8, places=2)
        self.assertAlmostEqual(result["X"][2].item(), 0.72, places=2)
        self.assertAlmostEqual(result["X"][3].item(), 0.608, places=2)

    def test_generating_initial_values(self):  #
        model = TimeseriesSampleGenerator(
            edges=[
                SampleEdge(
                    TimeAwareNodeReference("X", -1), TimeAwareNodeReference("X"), 0.9
                ),
                SampleEdge(
                    TimeAwareNodeReference("Y", -1), TimeAwareNodeReference("Y"), 0.9
                ),
                SampleEdge(
                    TimeAwareNodeReference("X", -1), TimeAwareNodeReference("Y"), 5
                ),
            ],
        )
        model._initial_distribution_fn = lambda x: x
        initial_values = model._calculate_initial_values()

        self.assertAlmostEqual(float(initial_values["X"] ** 2), 5.2630, places=0)
        self.assertAlmostEqual(float(initial_values["Y"] ** 2), 6602.2842, places=0)

    def test_generating_initial_values_additional_variable(self):
        model = TimeseriesSampleGenerator(
            edges=[
                SampleEdge(
                    TimeAwareNodeReference("X", -1), TimeAwareNodeReference("X"), 0.9
                ),
                SampleEdge(
                    TimeAwareNodeReference("Y", -1), TimeAwareNodeReference("Y"), 0.9
                ),
                SampleEdge(
                    TimeAwareNodeReference("Z", -1), TimeAwareNodeReference("Z"), 0.9
                ),
                SampleEdge(
                    TimeAwareNodeReference("X", -1), TimeAwareNodeReference("Y"), 5
                ),
                SampleEdge(
                    TimeAwareNodeReference("X", -1), TimeAwareNodeReference("Z"), 5
                ),
            ],
        )
        model._initial_distribution_fn = lambda x: x
        initial_values = model._calculate_initial_values()

        self.assertAlmostEqual(float(initial_values["X"] ** 2), 5.2630, places=0)
        # TODO: this is a bit of a hack, we have to fix numerical stability and then reduce the places
        self.assertAlmostEqual(float(initial_values["Y"] ** 2), 6602.2842, places=-2)
