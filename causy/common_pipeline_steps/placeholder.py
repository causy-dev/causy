import logging
from typing import Tuple, List, Generic, Optional

from causy.interfaces import (
    PipelineStepInterface,
    BaseGraphInterface,
    PipelineStepInterfaceType,
)
from causy.models import TestResultAction, TestResult
from causy.variables import StringVariable, IntegerVariable, FloatVariable, BoolVariable

logger = logging.getLogger(__name__)


class PlaceholderTest(
    PipelineStepInterface[PipelineStepInterfaceType], Generic[PipelineStepInterfaceType]
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "placeholder_str" in kwargs:
            self.placeholder_str = kwargs["placeholder_str"]

        if "placeholder_int" in kwargs:
            self.placeholder_int = kwargs["placeholder_int"]

        if "placeholder_float" in kwargs:
            self.placeholder_float = kwargs["placeholder_float"]

        if "placeholder_bool" in kwargs:
            self.placeholder_bool = kwargs["placeholder_bool"]

    placeholder_str: Optional[StringVariable] = "placeholder"
    placeholder_int: Optional[IntegerVariable] = 1
    placeholder_float: Optional[FloatVariable] = 1.0
    placeholder_bool: Optional[BoolVariable] = True

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
