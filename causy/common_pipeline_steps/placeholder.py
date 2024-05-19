import logging
from typing import Tuple, List, Generic

from causy.interfaces import (
    PipelineStepInterface,
    BaseGraphInterface,
    PipelineStepInterfaceType,
)
from causy.models import TestResultAction, TestResult

logger = logging.getLogger(__name__)


class PlaceholderTest(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    chunk_size_parallel_processing: int = 10
    parallel: bool = False

    def process(
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
