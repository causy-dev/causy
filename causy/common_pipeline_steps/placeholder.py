import logging
from typing import Tuple, List

from causy.interfaces import (
    PipelineStepInterface,
    TestResult,
    BaseGraphInterface,
    TestResultAction,
)

logger = logging.getLogger(__name__)


class PlaceholderTest(PipelineStepInterface):
    num_of_comparison_elements = 2
    chunk_size_parallel_processing = 10
    parallel = False

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface
    ) -> List[TestResult] | TestResult:
        """
        Placeholder test for testing purposes
        :param nodes:
        :param graph:
        :return:
        """
        logger.debug(f"PlaceholderTest {nodes}")
        return TestResult(u=None, v=None, action=TestResultAction.DO_NOTHING, data={})
