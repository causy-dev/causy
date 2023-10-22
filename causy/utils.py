import importlib
import logging

import torch
from sympy import transpose, Matrix, Symbol, shape
from scipy import stats as scipy_stats
import math
from statistics import correlation

logger = logging.getLogger(__name__)


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
    # TODO: rewrite this function with torch or remove it
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
            x[i] = b[i] / R[i, i]
        else:
            x_temp = 0
            for o, _ in enumerate(R[i, i + 1 :]):
                x_temp += x[i + 1 :][o] * R[i, i + 1 :][o]
            x[i] = (b[i] - x_temp) / R[i, i]
    return x


def get_regression_coefficients(x, Z):
    """
    :param x: list (length = # of samples), variable
    :param Z: list of lists (length of outer list = # of samples, length of inner list = # of variables in Z)
    :return: sympy matrix, regression coefficients from regressing x on Z
    """
    # TODO: rewrite this function with torch or remove it
    z_matrix = Matrix(Z)
    (n, m) = shape(z_matrix)
    logger.debug(f"(n,m)={(n, m)}")
    if m > n:
        raise ValueError(
            "Z must have at most as many rows as columns. (Otherwise you have more variables than samples - which seems to be the case here)"
        )
    Q, R = z_matrix.QRdecomposition()
    if shape(R) != (m, m):
        raise Exception("The matrix of data we regress on (Z) must have full rank.")
    q_transposed = transpose(Q)
    logger.debug(f"Q_transposed shape = {shape(q_transposed)}")
    logger.debug(f"R shape = {shape(R)}")
    x_matrix = Matrix(x)
    logger.debug(f"x shape = {shape(x_matrix)}")
    b = q_transposed @ x_matrix
    b_transposed = transpose(b)
    logger.debug(f"shape b={shape(b)}")
    regression_coefficients = backward_substituion(R, b_transposed, m - 1)
    return regression_coefficients


def get_residuals(x, Z):
    """
    :param x: list (length = # of samples), variable
    :param y: list (length = # of samples), variable
    :param Z: list of lists (length of outer list = # of samples, length of inner list = # of variables in Z)
    :return: residual – list (length = # of samples)

    CAUTION: Z must have full rank
    TODO: add optional check
    """
    # TODO: rewrite this function with torch or remove it

    n = len(x)
    res_x = Matrix(n, 1, x) - Matrix(Z) @ get_regression_coefficients(x, Z)
    return list(res_x)


def get_correlation(x, y, other_nodes):
    # TODO: rewrite this function with torch or remove it
    other_nodes_transposed = [list(i) for i in zip(*other_nodes)]
    residuals_x = get_residuals(x.values, other_nodes_transposed)
    residuals_y = get_residuals(y.values, other_nodes_transposed)
    corr = correlation(residuals_x, residuals_y)
    return corr


def get_t_and_critical_t(sample_size, nb_of_control_vars, par_corr, threshold):
    # TODO: rewrite this function with torch
    # check if we have to normalize data
    deg_of_freedom = sample_size - 2 - nb_of_control_vars
    if abs(round(par_corr, 4)) == 1:
        return (1, 0)
    critical_t = scipy_stats.t.ppf(1 - threshold / 2, deg_of_freedom)
    t = par_corr * math.sqrt(deg_of_freedom / (1 - par_corr**2))
    return t, critical_t


def serialize_module_name(cls):
    return f"{cls.__class__.__module__}.{cls.__class__.__name__}"


def load_pipeline_artefact_by_definition(step):
    name = step["name"]
    path = ".".join(name.split(".")[:-1])
    cls = name.split(".")[-1]
    st_function = importlib.import_module(path)
    st_function = getattr(st_function, cls)
    if not st_function:
        raise ValueError(f"{name} not found")

    if "params" not in step.keys():
        return st_function()
    else:
        return st_function(**step["params"])


def load_pipeline_steps_by_definition(steps):
    pipeline = []
    for step in steps:
        st_function = load_pipeline_artefact_by_definition(step)
        pipeline.append(st_function)
    return pipeline


def retrieve_edges(graph):
    edges = []
    for u in graph.edges:
        for v in graph.edges[u]:
            edges.append((u, v))
    return edges


def pearson_correlation(x, y):
    cov_xy = torch.mean((x - x.mean()) * (y - y.mean()))
    std_x = x.std()
    std_y = y.std()
    return cov_xy / (std_x * std_y)
