from abc import ABC, abstractmethod
from statistics import correlation, linear_regression, covariance
from typing import Tuple, List, Optional

import numpy as np

from utils import get_correlation

from interfaces import IndependenceTestInterface, BaseGraphInterface, NodeInterface, CorrelationTestResult, \
    CorrelationTestResultAction, AS_MANY_AS_FIELDS, ComparisonSettings


class CalculateCorrelations(IndependenceTestInterface):

    NUM_OF_COMPARISON_ELEMENTS = 2
    CHUNK_SIZE_PARALLEL_PROCESSING = 10000
    PARALLEL = False

    def test(self, edges: Tuple[str], graph: BaseGraphInterface) -> CorrelationTestResult:
        """
        Test if x and y are independent
        :param edges: the Edges to test
        :return: A CorrelationTestResult with the action to take
        """
        x = graph.nodes[edges[0]]
        y = graph.nodes[edges[1]]

        edge_value = graph.edge_value(graph.nodes[edges[0]], graph.nodes[edges[1]])
        edge_value["correlation"] = correlation(x.values, y.values)
        # edge_value["covariance"] = covariance(x.values, y.values)
        return CorrelationTestResult(x=x, y=y, action=CorrelationTestResultAction.UPDATE_EDGE, data=edge_value,)



class CorrelationCoefficientTest(IndependenceTestInterface):

    NUM_OF_COMPARISON_ELEMENTS = 2
    CHUNK_SIZE_PARALLEL_PROCESSING = 10000
    PARALLEL = True

    def test(self, edges: Tuple[str], graph: BaseGraphInterface) -> CorrelationTestResult:
        """
        Test if x and y are independent
        :param edges: the Edges to test
        :return: A CorrelationTestResult with the action to take
        """
        x = graph.nodes[edges[0]]
        y = graph.nodes[edges[1]]
        if abs(graph.edge_value(x,y)["correlation"]) <= self.threshold:
            return CorrelationTestResult(x=x, y=y, action=CorrelationTestResultAction.REMOVE_EDGE_UNDIRECTED, data={})

        return CorrelationTestResult(x=x, y=y, action=CorrelationTestResultAction.DO_NOTHING, data={})


class PartialCorrelationTest(IndependenceTestInterface):

    NUM_OF_COMPARISON_ELEMENTS = 3
    CHUNK_SIZE_PARALLEL_PROCESSING = 10000
    PARALLEL = True

    def test(self, edges: Tuple[str], graph: BaseGraphInterface) -> CorrelationTestResult:
        """
            Test if edges x,y are independent based on partial correlation with z as conditioning variable
            we use this test for all combinations of 3 nodes because it is faster than the extended test and we can
            use it to remove edges which are not independent and so reduce the number of combinations for the extended
            (See https://en.wikipedia.org/wiki/Partial_correlation#Using_recursive_formula)
            :param edges: the Edges to test
            :return: A CorrelationTestResult with the action to take
        """
        x: NodeInterface = graph.nodes[edges[0]]
        y: NodeInterface = graph.nodes[edges[1]]
        z: NodeInterface = graph.nodes[edges[2]]

        # Avoid division by zero
        if x is None or y is None or z is None:
            return CorrelationTestResult(x=x, y=y, action=CorrelationTestResultAction.DO_NOTHING)
        try:
            cor_xy = graph.edge_value(x, y)["correlation"]
            cor_xz = graph.edge_value(x, z)["correlation"]
            cor_yz = graph.edge_value(y, z)["correlation"]
        except KeyError:
            return CorrelationTestResult(x=x, y=y, action=CorrelationTestResultAction.DO_NOTHING)

        numerator = cor_xy - cor_xz * cor_yz
        denominator = ((1 - cor_xz ** 2) * (1 - cor_yz ** 2)) ** 0.5

        # Avoid division by zero
        if denominator == 0:
            return CorrelationTestResult(x=x, y=y, action=CorrelationTestResultAction.DO_NOTHING)

        # TODO: implement real independence test without scipy and numpy, see here:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        # or do a t test for the correlation coefficient

        if abs(numerator / denominator) <= self.threshold:
            return CorrelationTestResult(x=x, y=y, action=CorrelationTestResultAction.REMOVE_EDGE_UNDIRECTED, data={
                "separatedBy": [z]
            })

        return CorrelationTestResult(x=x, y=y, action=CorrelationTestResultAction.DO_NOTHING, data={})



class ExtendedPartialCorrelationTest(IndependenceTestInterface):

    NUM_OF_COMPARISON_ELEMENTS = ComparisonSettings(min=5, max=AS_MANY_AS_FIELDS)
    CHUNK_SIZE_PARALLEL_PROCESSING = 1

    def test(self, nodes: List[str], graph: BaseGraphInterface) -> CorrelationTestResult:
        """
        Test if edges x,y are independent based on partial correlation with z as conditioning variable
        we use this test for all combinations of more than 3 nodes because it is slower.

        """
        x = graph.nodes[nodes[0]]
        y = graph.nodes[nodes[1]]
        other_nodes = [graph.nodes[n].values for n in nodes[2:]]
        n = len(nodes)

        #insert function
        correlation_list = []
        for i in range(n):
            for j in range(i+1,n):
                x = graph.nodes[nodes[i]]
                y = graph.nodes[nodes[j]]
                exclude_indices = [i, j]
                other_nodes = [graph.nodes[n].values for idx, n in enumerate(nodes) if idx not in exclude_indices]
                corr = get_correlation(x,y,other_nodes)
                correlation_list.append(corr)

        # TODO  change to real independence test
        results = []
        for correlation in correlation_list:
            if abs(corr) <= self.threshold:
                results.append(CorrelationTestResult(x=x, y=y, action=CorrelationTestResultAction.REMOVE_EDGE_UNDIRECTED, data={
                    "separatedBy": other_nodes
                }))
        return results


class UnshieldedTriplesTest(IndependenceTestInterface):
    NUM_OF_COMPARISON_ELEMENTS = 2
    CHUNK_SIZE_PARALLEL_PROCESSING = 1000


    def test(self, edges: Tuple[str], graph: BaseGraphInterface) -> List[CorrelationTestResult]|CorrelationTestResult:
        # https://github.com/pgmpy/pgmpy/blob/1fe10598df5430295a8fc5cdca85cf2d9e1c4330/pgmpy/estimators/PC.py#L416

        x = graph.nodes[edges[0]]
        y = graph.nodes[edges[1]]

        if graph.edge_exists(x, y):
            return CorrelationTestResult(x=x, y=y, action=CorrelationTestResultAction.DO_NOTHING, data={})

        potential_zs = set(graph.edges[x].keys()).intersection(set(graph.edges[y].keys()))

        for z in potential_zs:
            separators = graph.retrieve_edge_history(x, y, CorrelationTestResultAction.REMOVE_EDGE_UNDIRECTED)

            if z not in separators:
                return [CorrelationTestResult(x=z, y=x, action=CorrelationTestResultAction.REMOVE_EDGE_DIRECTED, data={}),
                        CorrelationTestResult(x=z, y=y, action=CorrelationTestResultAction.REMOVE_EDGE_DIRECTED, data={})]

        return CorrelationTestResult(x=x, y=y, action=CorrelationTestResultAction.DO_NOTHING, data={})


class ExtendedPartialCorrelationTest2(IndependenceTestInterface):

    NUM_OF_COMPARISON_ELEMENTS = ComparisonSettings(min=4, max=AS_MANY_AS_FIELDS)
    CHUNK_SIZE_PARALLEL_PROCESSING = 50
    PARALLEL = False


    def test(self, nodes: List[str], graph: BaseGraphInterface) -> CorrelationTestResult:

        covariance_matrix = [[None for _ in range(len(nodes))] for _ in range(len(nodes))]
        for i in range(len(nodes)):
            for k in range(i, len(nodes)):
                if covariance_matrix[i][k] is None:
                    covariance_matrix[i][k] = covariance(graph.nodes[nodes[i]].values, graph.nodes[nodes[k]].values)
                    covariance_matrix[k][i] = covariance_matrix[i][k]

        cov_matrix = np.array(covariance_matrix)
        print(cov_matrix)
        inverse_cov_matrix = np.linalg.inv(cov_matrix)
        n = len(inverse_cov_matrix)
        diagonal = np.diagonal(inverse_cov_matrix)
        diagonal_matrix = np.zeros((n, n))
        np.fill_diagonal(diagonal_matrix, diagonal)
        print(diagonal_matrix)
        print(inverse_cov_matrix)
        print(diagonal_matrix[3][3] == inverse_cov_matrix[3][3])
        helper = np.dot(np.sqrt(diagonal_matrix), inverse_cov_matrix)
        partial_correlation_coefficients = np.dot(helper, np.sqrt(diagonal_matrix))
        print("partial_correlation_coefficients")
        print(partial_correlation_coefficients)
        par_corr_xy = partial_correlation_coefficients[1][0]

class PlaceholderTest(IndependenceTestInterface):
    NUM_OF_COMPARISON_ELEMENTS = 2
    CHUNK_SIZE_PARALLEL_PROCESSING = 10
    PARALLEL = False


    def test(self, edges: Tuple[str], graph: BaseGraphInterface) -> List[CorrelationTestResult]|CorrelationTestResult:
        print("PlaceholderTest")
        return CorrelationTestResult(x=None, y=None, action=CorrelationTestResultAction.DO_NOTHING, data={})
