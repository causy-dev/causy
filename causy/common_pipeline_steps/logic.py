import time
from typing import Optional, Generic


from causy.interfaces import (
    LogicStepInterface,
    BaseGraphInterface,
    GraphModelInterface,
    ExitConditionInterface,
    LogicStepInterfaceType,
)

from causy.models import ActionHistoryStep


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
        steps = []
        loop_started = time.time()
        actions_taken = None
        while not self.exit_condition(
            graph=graph,
            graph_model_instance_=graph_model_instance_,
            actions_taken=actions_taken,
            iteration=n,
        ):
            actions_taken = []
            for pipeline_step in self.pipeline_steps:
                started = time.time()
                result = graph_model_instance_.execute_pipeline_step(pipeline_step)
                steps.append(
                    ActionHistoryStep(
                        name=pipeline_step.name,
                        actions=result,
                        duration=time.time() - started,
                    )
                )
                actions_taken.extend(result)
            n += 1
        return ActionHistoryStep(
            name=self.name,
            steps=steps,
            duration=time.time() - loop_started,
        )


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
        steps = []
        loop_started = time.time()
        for pipeline_step in self.pipeline_steps:
            started = time.time()
            result = graph_model_instance_.execute_pipeline_step(
                pipeline_step, apply_to_graph=False
            )
            steps.append(
                ActionHistoryStep(
                    name=pipeline_step.name,
                    actions=result,
                    duration=time.time() - started,
                )
            )
            actions.extend(result)

        graph_model_instance_._take_action(actions)

        return ActionHistoryStep(
            name=self.name,
            steps=steps,
            duration=time.time() - loop_started,
        )
