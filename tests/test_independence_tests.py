import random
import unittest

import independence_tests
from graph import UndirectedGraph

from utils import sum_lists
from independence_tests import ExtendedPartialCorrelationTest

class IndependenceTestTestCase(unittest.TestCase):

    def test_correlation_coefficient_standard_model(self):
        tst = independence_tests.CorrelationCoefficientTest()

        n = 1000
        x = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        noise_y = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        y = sum_lists([5*x_val for x_val in x], noise_y)
        self.assertEqual(
            tst(x, y, UndirectedGraph()),
            False
        )

    def test_correlation_coefficient_standard_model_weak_effect(self):
        tst = independence_tests.CorrelationCoefficientTest()

        n = 1000
        x = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        noise_y = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        y = sum_lists([0.3*x_val for x_val in x], noise_y)
        self.assertEqual(
            tst(x, y, UndirectedGraph()),
            False
        )

    def test_correlation_of_noise(self):
        tst = independence_tests.CorrelationCoefficientTest(threshold=0.2)
        n = 1000
        x = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        y = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        self.assertEqual(
            tst([x, y], UndirectedGraph()),
            True
        )

    def test_extended_partial_correlation_test(self):
        tst = ExtendedPartialCorrelationTest(threshold=0.2)
        n = 1000
        x = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        noise_y = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        noise_z1 = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        noise_z2 = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        noise_z3 = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
        z1 = sum_lists([3 * x_val for x_val in x], noise_z1)
        z2 = sum_lists([3 * x_val for x_val in x], noise_z2)
        z3 = sum_lists([3 * x_val for x_val in x], noise_z3)
        y = sum_lists([3*z1_val for z1_val in z1], [3*z2_val for z2_val in z2], [3*z3_val for z3_val in z3], noise_y)
        self.assertEqual(
            tst([x, y, z1, z2, z3], UndirectedGraph()),
            True
        )

if __name__ == '__main__':
    unittest.main()

