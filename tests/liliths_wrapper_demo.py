from typing import Generic, List, Optional

from causy.causal_discovery.constraint.orientation_rules.pc import FurtherOrientQuadrupleTest, ColliderTest
from causy.interfaces import PipelineStepInterface, PipelineStepInterfaceType, BaseGraphInterface, TestResultInterface, \
    logger


def actual_fn(a,b):
    return a + b


def wrapper_fn(a,b):
    result = actual_fn(a, b)
    return result * -1



class CustomOrientationRule(PipelineStepInterface[PipelineStepInterfaceType],
                            Generic[PipelineStepInterfaceType]):

    _inner_step_cls = ColliderTest

    def __int__(self, *args, **kwargs):
        self.inner_step = self._inner_step_cls(*args, **kwargs)
        super().__init__(*args, **kwargs)

    def process(self, nodes: List[str], graph: BaseGraphInterface,
                unapplied_actions: Optional[List[TestResultInterface]] = None) -> Optional[TestResultInterface]:
        result = self.inner_step.process(nodes, graph, unapplied_actions=unapplied_actions)

        # get all unshielded triples


        for proposed_action in result.all_proposed_actions:
            if "separatedBy" in proposed_action.data:
        return result

    def __call__(
        self,
        nodes: List[str],
        graph: BaseGraphInterface,
        unapplied_actions: Optional[List[TestResultInterface]] = None,
    ) -> Optional[TestResultInterface]:
        if self.needs_unapplied_actions and unapplied_actions is None:
            logger.warn(
                f"Pipeline step {self.name} needs unapplied actions but none were provided"
            )
        elif self.needs_unapplied_actions and unapplied_actions is not None:
            return self.process(nodes, graph, unapplied_actions)

        return self.process(nodes, graph)



