from typing import Optional, List, Union, Dict, Any, Generic

from pydantic import BaseModel

from causy.interfaces import (
    LogicStepInterface,
    BaseGraphInterface,
    GraphModelInterface,
    PipelineStepInterface,
    ExitConditionInterface,
    PipelineStepInterfaceType,
    LogicStepInterfaceType,
)
from causy.graph_utils import (
    load_pipeline_artefact_by_definition,
    load_pipeline_steps_by_definition,
)


class Loop(LogicStepInterface[LogicStepInterfaceType], Generic[LogicStepInterfaceType]):
    """
    A loop which executes a list of pipeline_steps until the exit_condition is met.
    """

    exit_condition: Optional[ExitConditionInterface] = None

    def execute(
        self, graph: BaseGraphInterface, graph_model_instance_: GraphModelInterface
    ):
        """
        Executes the loop til self.exit_condition is met
        :param graph:
        :param graph_model_instance_:
        :return:
        """
        n = 0
        steps = None
        while not self.exit_condition(
            graph=graph,
            graph_model_instance_=graph_model_instance_,
            actions_taken=steps,
            iteration=n,
        ):
            steps = []
            for pipeline_step in self.pipeline_steps:
                result = graph_model_instance_.execute_pipeline_step(pipeline_step)
                steps.extend(result)
            n += 1


class ApplyActionsTogether(
    LogicStepInterface[LogicStepInterfaceType], Generic[LogicStepInterfaceType]
):
    """
    A logic step which collects all actions and only takes them at the end of the pipeline
    """

    def execute(
        self, graph: BaseGraphInterface, graph_model_instance_: GraphModelInterface
    ):
        """
        Executes the loop til self.exit_condition is met
        :param graph:
        :param graph_model_instance_:
        :return:
        """
        actions = []
        for pipeline_step in self.pipeline_steps:
            result = graph_model_instance_.execute_pipeline_step(pipeline_step)
            actions.extend(result)

        graph_model_instance_._take_action(actions)
