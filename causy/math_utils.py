import logging
from scipy import stats as scipy_stats
import math


logger = logging.getLogger(__name__)


def sum_lists(*lists):
    """
    :param lists: lists of numbers
    :return: list (sum of lists)
    """
    return list(map(sum, zip(*lists)))


def get_t_and_critical_t(sample_size, nb_of_control_vars, par_corr, threshold):
    # TODO: rewrite this function with torch
    # check if we have to normalize data
    deg_of_freedom = sample_size - 2 - nb_of_control_vars
    if abs(round(par_corr, 4)) == 1:
        return (1, 0)
    critical_t = scipy_stats.t.ppf(1 - threshold / 2, deg_of_freedom)
    t = par_corr * math.sqrt(deg_of_freedom / (1 - par_corr**2))
    return t, critical_t
