import unittest

from sympy import transpose, Matrix, solve_linear_system, symbols

from utils import (
    backward_substituion,
    get_regression_coefficients,
    get_residuals,
    extended_partial_correlation_2_test,
)


class UtilsTestCase(unittest.TestCase):
    def test_QR_decomposition(self):
        x = [1, 2, 3]
        Z = [[12, -51], [6, 167], [-4, 21]]
        n = len(x)
        M = Matrix(Z)
        Q, R = M.QRdecomposition()
        print(f"Q shape = {Q.shape}")
        print(f"R shape = {R.shape}")
        # print(f"R ist {R}")
        self.assertTrue(False)

    def test_backward_substitution(self):
        R = Matrix([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
        b = Matrix([6, 15, 24])
        n = 2
        self.assertEqual(
            backward_substituion(R, b, n), Matrix([[-7 / 2], [-5 / 4], [4]])
        )

    def test_regression_coefficients(self):
        x = [1, 2, 3]
        Z = [[12, -51], [6, 167], [-4, 21]]
        print(get_regression_coefficients(x, Z))
        self.assertTrue(False)

    def test_regression_coefficients_2(self):
        x = [1, 2, 3, 4, 5]
        Z = [[2, 3, 5], [3, 2, 2], [-16, 3, 1], [4, 3, 12], [5, 7, 120]]
        print(get_regression_coefficients(x, Z))
        self.assertTrue(False)

    def test_get_residuals(self):
        x = [1, 2, 3, 4, 5]
        Z = [[2, 3, 5], [3, 2, 2], [-16, 3, 1], [4, 3, 12], [5, 7, 120]]
        print(f"residuals are {get_residuals(x, Z)}")
        self.assertTrue(False)

    def test_shapes(self):
        x = [1, 2, 3, 4, 5]
        y = [-3, 2, 4, -6, 1]
        Z = [[2, 3, 5], [3, 2, 2], [-16, 3, 1], [4, 3, 12], [5, 7, 120]]
        print(extended_partial_correlation_2_test(x, y, Z))


if __name__ == "__main__":
    unittest.main()
