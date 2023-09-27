import operator
import numpy as np
from sympy import transpose, Matrix, solve_linear_system, symbols, Symbol, shape


def sum_lists(*lists):
    """
   :param lists: lists of numbers
   :return: list (sum of lists)
   """
    return list(map(sum, zip(*lists)))


def backward_substituion(R, b, n):
    """
   :param R: sympy matrix, nxn-dimensional upper triangular matrix (from QR decomposition)
   :param b: sympy matrix, n-dimensional vector (b=Q^Tx)
   :param n: int, dimension
   :return: Matrix, n-dimensional regression coefficient vector
   """

    # Define the symbolic variables
    my_symbols = []
    for z in range(n + 1):
        locals()[f"x{z}"] = Symbol(f"x{z}")
        my_symbols.append(locals()[f"x{z}"])

    # Initialize the solution vector x
    x = Matrix([*my_symbols])

    # Solve for x using back-substitution
    for i in range(n, -1, -1):
        if i == n:
            print(b)
            print(i)
            x[i] = b[i] / R[i, i]
        else:
            x_temp = 0
            for o, _ in enumerate(R[i, i + 1:]):
                x_temp += x[i + 1:][o] * R[i, i + 1:][o]
            x[i] = (b[i] - x_temp) / R[i, i]
    return x


def get_regression_coefficients(x, Z):
    """
   :param x: list (length = # of samples), variable
   :param Z: list of lists (length of outer list = # of samples, length of inner list = # of variables in Z)
   :return: sympy matrix, regression coefficients from regressing x on Z
   """
    z_matrix = Matrix(Z)
    #print(z_matrix)
    (n, m) = shape(z_matrix)
    print(f"(n,m)={(n, m)}")
    if m > n:
        raise ValueError(
            "Z must have at most as many rows as columns. (Otherwise you have more variables than samples - which seems to be the case here)")
    Q, R = z_matrix.QRdecomposition()
    if not shape(R) == (m,m):
        raise Exception("The matrix of data we regress on (Z) must have full rank.")
    #print(R)
    #print(Q)
    q_transposed = transpose(Q)
    print(f"Q_transposed shape = {shape(q_transposed)}")
    print(f"R shape = {shape(R)}")
    x_matrix = Matrix(x)
    print(f"x shape = {shape(x_matrix)}")
    b = q_transposed @ x_matrix
    b_transposed = transpose(b)
    print(f"shape b={shape(b)}")
    regression_coefficients = backward_substituion(R, b_transposed, m - 1)
    return regression_coefficients

def get_residuals(x,Z):
    """
    :param x: list (length = # of samples), variable
    :param y: list (length = # of samples), variable
    :param Z: list of lists (length of outer list = # of samples, length of inner list = # of variables in Z)
    :return: residual – list (length = # of samples)

    CAUTION: Z must have full rank
    TODO: add optional check
    """
    n = len(x)
    res_x = Matrix(n,1,x) - Matrix(Z) @ get_regression_coefficients(x, Z)
    return list(res_x)

def extended_partial_correlation_2_test(edge_values):

    # just to check if this works and delivers same results as current implementation
    # delete later and implement differently

    # Create matrix


    # Step 1: Compute the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)

    # Step 2: Calculate the inverse of the covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Step 3: Calculate partial correlations
    num_vars = cov_matrix.shape[0]

    partial_corr_matrix = np.zeros((num_vars, num_vars))

    for i in range(num_vars):
        for j in range(num_vars):
            if i != j:
                partial_corr_matrix[i, j] = -inv_cov_matrix[i, j] / np.sqrt(inv_cov_matrix[i, i] * inv_cov_matrix[j, j])

    # Print the partial correlation matrix
    print("Partial Correlation Matrix:")
    print(partial_corr_matrix)
    pass


def get_correlation(x,y,other_nodes)
    other_nodes_transposed = [list(i) for i in zip(*other_nodes)]
    if x is None or y is None:
        return CorrelationTestResult(x=x, y=y, action=CorrelationTestResultAction.DO_NOTHING)
    residuals_x = get_residuals(x.values, other_nodes_transposed)
    residuals_y = get_residuals(y.values, other_nodes_transposed)
    corr = correlation(residuals_x, residuals_y)
    return corr
