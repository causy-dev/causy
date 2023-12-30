import logging
from typing import Tuple, List

from causy.interfaces import (
    PipelineStepInterface,
    TestResult,
    BaseGraphInterface,
    TestResultAction,
)

import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


class PlaceholderTest(PipelineStepInterface):
    num_of_comparison_elements = 2
    chunk_size_parallel_processing = 10
    parallel = False

    def test(
        self, nodes: Tuple[str], graph: BaseGraphInterface, result_queue: mp.Queue
    ):
        """
        Placeholder test for testing purposes
        :param nodes:
        :param graph:
        :param result_queue:
        :return:
        """
        logger.debug(f"PlaceholderTest {nodes}")
        result_queue.put(
            TestResult(x=None, y=None, action=TestResultAction.DO_NOTHING, data={})
        )
