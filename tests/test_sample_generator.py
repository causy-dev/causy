import torch
import numpy as np

from causy.sample_generator import (
    TimeseriesSampleGenerator,
    SampleEdge,
    IIDSampleGenerator,
    TimeAwareNodeReference,
    NodeReference,
)

from tests.utils import CausyTestCase


class TimeseriesSampleGeneratorTest(CausyTestCase):
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

        self.assertAlmostEqual(float(initial_values["X"] ** 2), 5.2630, places=2)
        # TODO: this is a bit of a hack, we have to fix numerical stability and then reduce the places
        self.assertAlmostEqual(float(initial_values["Y"] ** 2), 6602.2842, places=-2)

    def test_generate_covariance_matrix_1(self):
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

        coefficient_matrix = model._generate_coefficient_matrix()
        cov_matrix = model._generate_covariance_matrix(
            coefficient_matrix, list(coefficient_matrix.size())[0]
        )

        self.assertAlmostEqual(cov_matrix[0][0].item(), 5.3, places=1)
        self.assertAlmostEqual(cov_matrix[1][1].item(), 6602.2, places=0)
        self.assertAlmostEqual(cov_matrix[0][1].item(), 124.65, places=1)
        self.assertAlmostEqual(cov_matrix[1][0].item(), 124.65, places=1)

    def test_generate_covariance_matrix_2(self):
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
                    TimeAwareNodeReference("Y", -1), TimeAwareNodeReference("Z"), 7
                ),
            ],
        )

        coefficient_matrix = model._generate_coefficient_matrix()
        cov_matrix = model._generate_covariance_matrix(
            coefficient_matrix, list(coefficient_matrix.size())[0]
        )

        self.assertAlmostEqual(cov_matrix[0][0].item(), 5.3, places=1)
        self.assertAlmostEqual(cov_matrix[1][1].item(), 6602.2, places=0)
        self.assertAlmostEqual(cov_matrix[0][1].item(), 124.6, places=1)
        self.assertAlmostEqual(cov_matrix[1][0].item(), 124.6, places=1)

    def test_generate_covariance_matrix_3(self):
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

        coefficient_matrix = model._generate_coefficient_matrix()
        cov_matrix = model._generate_covariance_matrix(
            coefficient_matrix, list(coefficient_matrix.size())[0]
        )

        self.assertAlmostEqual(cov_matrix[0][0].item(), 5.3, places=1)
        self.assertAlmostEqual(cov_matrix[1][1].item(), 6602.2, places=0)
        self.assertAlmostEqual(cov_matrix[0][1].item(), 124.6, places=1)
        self.assertAlmostEqual(cov_matrix[1][0].item(), 124.6, places=1)
        self.assertAlmostEqual(cov_matrix[2][0].item(), 124.6, places=1)
        self.assertAlmostEqual(cov_matrix[0][2].item(), 124.6, places=1)
        self.assertAlmostEqual(cov_matrix[2][2].item(), 6602.2, places=0)

    def test_generate_coefficient_matrix(self):
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
        coefficient_matrix = model._generate_coefficient_matrix()
        self.assertEqual(len(coefficient_matrix), 2)
        self.assertEqual(len(coefficient_matrix[0]), 2)
        self.assertEqual(coefficient_matrix[0][0], 0.9)
        self.assertEqual(coefficient_matrix[1][1], 0.9)
        self.assertEqual(coefficient_matrix[1][0], 5)
        self.assertEqual(coefficient_matrix[0][1], 0)

    def test_generate_coefficient_matrix_2(self):
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
                    TimeAwareNodeReference("X", -1), TimeAwareNodeReference("Z"), 7
                ),
            ],
        )
        coefficient_matrix = model._generate_coefficient_matrix()
        self.assertEqual(len(coefficient_matrix), 3)
        self.assertEqual(len(coefficient_matrix[0]), 3)
        self.assertEqual(coefficient_matrix[0][0], 0.9)
        self.assertEqual(coefficient_matrix[1][1], 0.9)
        self.assertEqual(coefficient_matrix[1][0], 5)
        self.assertEqual(coefficient_matrix[0][1], 0)
        self.assertEqual(coefficient_matrix[2][0], 7)
        self.assertEqual(coefficient_matrix[0][2], 0)

    def test_vectorize_identity_block(self):
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
        n = list(model._generate_coefficient_matrix().size())[0]
        numpy_vec_cov_matrix = np.eye(n).flatten()
        torch_vec_cov_matrix = model._vectorize_identity_block(n)
        # test that numpy array and torch tensor have the same values
        numpy_list = numpy_vec_cov_matrix.tolist()
        torch_list = torch_vec_cov_matrix.tolist()
        self.assertListEqual(numpy_list, torch_list)

    def test_vectorize_identity_block_2(self):
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
                    TimeAwareNodeReference("X", -1), TimeAwareNodeReference("Z"), 7
                ),
            ],
        )
        n = list(model._generate_coefficient_matrix().size())[0]
        numpy_vec_cov_matrix = np.eye(n).flatten()
        torch_vec_cov_matrix = model._vectorize_identity_block(n)
        # test that numpy array and torch tensor have the same values
        numpy_list = numpy_vec_cov_matrix.tolist()
        torch_list = torch_vec_cov_matrix.tolist()
        self.assertListEqual(numpy_list, torch_list)

    # test if numerical errors arise from using torch in comparision to numpy
    def test_reshape_numpy_vs_torch(self):
        numpy_vector = np.array([1, 2, 3, 4])
        torch_vector = torch.tensor([1, 2, 3, 4])
        numpy_vector_reshaped = numpy_vector.reshape(2, 2)
        torch_vector_reshaped = torch_vector.reshape(2, 2)
        for i in range(2):
            for j in range(2):
                self.assertEqual(
                    numpy_vector_reshaped[i][j], torch_vector_reshaped[i][j].item()
                )

    def test_matmul_torch_numpy(self):
        numpy_matrix = np.array(
            [[1, 2, 3, 4], [3, 4, 5, 8], [2, 4, 5, 8], [1, 1, 1, 1]]
        )
        torch_matrix = torch.tensor(
            [[1, 2, 3, 4], [3, 4, 5, 8], [2, 4, 5, 8], [1, 1, 1, 1]]
        )
        numpy_vector = np.array([1, 2, 3, 4])
        torch_vector = torch.tensor([1, 2, 3, 4])
        torch_matrix_vector_multiplication = torch.matmul(
            torch_matrix, torch_vector
        ).tolist()
        numpy_matrix_vector_multiplication = np.matmul(
            numpy_matrix, numpy_vector
        ).tolist()
        self.assertListEqual(
            torch_matrix_vector_multiplication, numpy_matrix_vector_multiplication
        )

    def test_kronecker_product_numpy_torch(self):
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
        coefficient_matrix = model._generate_coefficient_matrix()
        torch_kron_product = torch.kron(coefficient_matrix, coefficient_matrix).tolist()
        coefficient_matrix_numpy = model._generate_coefficient_matrix().numpy()
        numpy_kron_product = np.kron(
            coefficient_matrix_numpy, coefficient_matrix_numpy
        ).tolist()
        self.assertListEqual(torch_kron_product, numpy_kron_product)

    def test_inverse_of_difference_numpy_torch(self):
        numpy_identity = np.identity(4)
        numpy_matrix = np.array(
            [[1, 2, 3, 4], [3, 4, 5, 8], [2, 4, 5, 8], [1, 1, 1, 1]]
        )
        torch_identity = torch.eye(4)
        torch_matrix = torch.tensor(
            [[1, 2, 3, 4], [3, 4, 5, 8], [2, 4, 5, 8], [1, 1, 1, 1]]
        )
        numpy_inverse = np.linalg.inv(numpy_identity - numpy_matrix).flatten().tolist()
        helper_list_of_tensors = torch.linalg.pinv(torch_identity - torch_matrix)
        torch_inverse = [
            float(element) for sublist in helper_list_of_tensors for element in sublist
        ]
        # allow very small numerical errors
        for i in range(4**2):
            self.assertAlmostEqual(numpy_inverse[i], torch_inverse[i], 5)
