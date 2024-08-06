from abc import ABC
from typing import List, TypeVar, Generic

import torch
from pydantic import BaseModel, computed_field

from causy.graph import Graph, Node, logger
from causy.graph_utils import serialize_module_name
from causy.math_utils import get_t_and_critical_t


def invert_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if torch.det(matrix) == 0:
        return torch.linalg.pinv(matrix)
    else:
        return torch.inverse(matrix)


ConditionalIndependenceTestInterfaceType = TypeVar(
    "ConditionalIndependenceTestInterfaceType"
)


class ConditionalIndependenceTestInterface(
    ABC, BaseModel, Generic[ConditionalIndependenceTestInterfaceType]
):
    @computed_field
    @property
    def name(self) -> str:
        return serialize_module_name(self)

    @staticmethod
    def calculate_correlation(x: Node, y: Node, z: List[Node]) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def test(graph: Graph, x: str, y: str, z: List[str], threshold: float) -> bool:
        raise NotImplementedError


class PearsonStudentsTTest(
    ConditionalIndependenceTestInterface[ConditionalIndependenceTestInterfaceType],
    Generic[ConditionalIndependenceTestInterfaceType],
):
    @staticmethod
    def calculate_correlation(x: Node, y: Node, z: List[Node]) -> torch.Tensor:
        """
        Calculate the correlation between two nodes x and y given a list of control variables z.
        It returns a tensor with the t-value and the critical t-value.
        It uses the Pearson's t-test for the correlation coefficient.
        :param x:
        :param y:
        :param z:
        :param threshold:
        :return:
        """

        if len(z) == 0:
            cov_xy = torch.mean(
                (x.values - x.values.mean()) * (y.values - y.values.mean())
            )
            std_x = x.values.std(unbiased=False)
            std_y = y.values.std(unbiased=False)
            pearson_correlation = cov_xy / (std_x * std_y)

            correlation = pearson_correlation.item()

            # Clamp the correlation to -1 and 1 to avoid numerical errors
            if correlation < -1:
                correlation = -1
            elif correlation > 1:
                correlation = 1

            return torch.tensor(correlation)

        cov_matrix = torch.cov(
            torch.stack([x.values, y.values, *[zi.values for zi in z]])
        )
        # check if the covariance matrix is ill-conditioned
        inverse_cov_matrix = invert_matrix(cov_matrix)

        n = inverse_cov_matrix.size(0)
        diagonal = torch.diag(inverse_cov_matrix)
        diagonal_matrix = torch.zeros((n, n), dtype=torch.float64)
        for i in range(n):
            diagonal_matrix[i, i] = diagonal[i]

        helper = torch.mm(torch.sqrt(diagonal_matrix), inverse_cov_matrix)
        precision_matrix = torch.mm(helper, torch.sqrt(diagonal_matrix))

        return (-1 * precision_matrix[0][1]) / torch.sqrt(
            precision_matrix[0][0] * precision_matrix[1][1]
        )

    @staticmethod
    def test(graph: Graph, x: str, y: str, z: List[str], threshold: float) -> bool:
        """
        :param graph:
        :param x:
        :param y:
        :param z:
        :return:
        """
        x = graph.nodes[x]
        y = graph.nodes[y]
        z = [graph.nodes[zi] for zi in z]

        res = None

        if len(z) == 0:
            edge = graph.edge_value(x, y)
            if edge is not None and "correlation" in edge:
                res = torch.tensor(edge["correlation"])

        if res is None:
            res = PearsonStudentsTTest.calculate_correlation(x, y, z)

        sample_size = len(x.values)
        nb_of_control_vars = len(z)

        # prevent math domain error
        try:
            t, critical_t = get_t_and_critical_t(
                sample_size, nb_of_control_vars, res.item(), threshold
            )
        except ValueError:
            logger.warning(
                "Math domain error. The covariance matrix is ill-conditioned. The precision matrix is not reliable."
            )
            return None

        return abs(t) < critical_t


class FishersZTest(
    ConditionalIndependenceTestInterface[ConditionalIndependenceTestInterfaceType],
    Generic[ConditionalIndependenceTestInterfaceType],
):
    @staticmethod
    def calculate_correlation(x: Node, y: Node, z: List[Node]) -> torch.Tensor:
        if len(z) == 0:
            r = torch.corrcoef(torch.stack([x.values, y.values]))[0, 1]
        else:
            sub_corr = torch.corrcoef(
                torch.stack([x.values, y.values, *[zi.values for zi in z]])
            )[0, 1]
            r = invert_matrix(sub_corr)
            r = -1 * r[0, 1] / torch.sqrt(abs(r[0, 0] * r[1, 1]))

        cut_at = torch.tensor(0.99999)
        r = torch.min(cut_at, torch.max(-1 * cut_at, r))  # make r between -1 and 1

        res = torch.sqrt(
            torch.tensor(len(x.values)) - torch.tensor(len(z)) - 3
        ) * torch.atanh(r)
        p = 2 * (1 - torch.distributions.Normal(0, 1).cdf(res))
        return p

    @staticmethod
    def test(graph: Graph, x: str, y: str, z: List[str], threshold: float) -> bool:
        """
        :param graph:
        :param x:
        :param y:
        :param z:
        :return:
        """
        x = graph.nodes[x]
        y = graph.nodes[y]
        z = [graph.nodes[zi] for zi in z]

        p = FishersZTest.calculate_correlation(x, y, z)
        p = p.item()
        return p < threshold


class ChiSquareTest(
    ConditionalIndependenceTestInterface[ConditionalIndependenceTestInterfaceType],
    Generic[ConditionalIndependenceTestInterfaceType],
):
    @staticmethod
    def test(graph: Graph, x: str, y: str, z: List[str], threshold: float) -> bool:
        return True


class G2Test(
    ConditionalIndependenceTestInterface[ConditionalIndependenceTestInterfaceType],
    Generic[ConditionalIndependenceTestInterfaceType],
):
    @staticmethod
    def test(graph: Graph, x: str, y: str, z: List[str], threshold: float) -> bool:
        return True
