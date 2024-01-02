from causy.interfaces import ExitConditionInterface


class ExitOnNoActions(ExitConditionInterface):
    def check(self, graph, graph_model_instance_, actions_taken, iteration) -> bool:
        """
        Check if there are no actions taken in the last iteration and if so, break the loop
        If it is the first iteration, do not break the loop (we need to execute the first step)
        :param graph: the graph
        :param graph_model_instance_: the graph model instance
        :param actions_taken: the actions taken in the last iteration
        :param iteration: iteration number
        :return: True if you want to break an iteration, False otherwise
        """
        if iteration > 0:
            return True if len(actions_taken) == 0 else False
        else:
            return False
