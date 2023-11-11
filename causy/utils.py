import importlib
import logging
from typing import List, Tuple

import torch
from scipy import stats as scipy_stats
import math


logger = logging.getLogger(__name__)


def sum_lists(*lists):
    """
    :param lists: lists of numbers
    :return: list (sum of lists)
    """
    return list(map(sum, zip(*lists)))


def custom_round_single_number(number: torch.Tensor, precision: int = 4):
    """
    custom round function for a single number as torch does not support round on all devices
    :param number:
    :param precision:
    :return:
    """
    factor = 10**precision
    rounded_number = torch.floor(number * factor + 0.5) / factor
    return rounded_number


def get_t_and_critical_t(
    sample_size: int, nb_of_control_vars: int, par_corr: torch.Tensor, threshold: float
):
    """
    Returns the t and critical t values for a given sample size, number of control variables, partial correlation and threshold
    :param sample_size:
    :param nb_of_control_vars:
    :param par_corr: the partial correlation as a tensor
    :param threshold:
    :return:
    """
    # TODO: rewrite this function with torch
    # check if we have to normalize data
    deg_of_freedom = sample_size - 2 - nb_of_control_vars
    if torch.eq(torch.abs(custom_round_single_number(par_corr, precision=4)), 1).any():
        return (1, 0)
    critical_t = scipy_stats.t.ppf(1 - threshold / 2, deg_of_freedom)

    t = par_corr.mul(torch.sqrt(deg_of_freedom / (1 - torch.pow(par_corr, 2))))
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


def retrieve_edges(graph) -> List[Tuple[str, str]]:
    """
    Returns a list of edges from the graph
    :param graph: a graph
    :return: a list of edges
    """
    edges = []
    for u in graph.edges:
        for v in graph.edges[u]:
            edges.append((u, v))
    return edges


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Returns the pearson correlation coefficient between x and y
    :param x: a tensor
    :param y: a tensor
    :return: the correlation coefficient
    """
    cov_xy = torch.mean((x - x.mean()) * (y - y.mean()))
    std_x = x.std(unbiased=False)
    std_y = y.std(unbiased=False)
    return cov_xy / (std_x * std_y)
