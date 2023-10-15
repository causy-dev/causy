import unittest

from sympy import Matrix

from causy.utils import backward_substituion


class UtilsTestCase(unittest.TestCase):
    def test_QR_decomposition(self):
        Z = [[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]]
        M = Matrix(Z)
        Q, R = M.QRdecomposition()
        self.assertAlmostEqual(R, Matrix([[2, 3, 2], [0, 5, -2], [0, 0, 4]]))

    def test_backward_substitution(self):
        R = Matrix([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
        b = Matrix([6, 15, 24])
        n = 2
        self.assertEqual(
            backward_substituion(R, b, n), Matrix([[-7 / 2], [-5 / 4], [4]])
        )


if __name__ == "__main__":
    unittest.main()
