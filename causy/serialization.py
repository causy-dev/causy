from typing import Dict, Any

from pydantic import parse_obj_as

from causy.graph_utils import serialize_module_name, load_pipeline_steps_by_definition


def serialize_algorithm(model, algorithm_name: str = None):
    """Serialize the model into a dictionary."""

    if algorithm_name:
        model.algorithm.name = algorithm_name
    return model.algorithm.model_dump()


def load_algorithm_from_specification(algorithm_dict: Dict[str, Any]):
    """Load the model from a dictionary."""
    algorithm_dict["pipeline_steps"] = load_pipeline_steps_by_definition(
        algorithm_dict["pipeline_steps"]
    )
    from causy.interfaces import CausyAlgorithm

    return parse_obj_as(CausyAlgorithm, algorithm_dict)


class SerializeMixin:
    """Mixin class for serializing and deserializing graph steps."""

    def _serialize_object(self, obj):
        """Serialize the object into a dictionary."""
        result = {}
        for attr in [
            attr
            for attr in dir(obj)
            if not attr.startswith("__") and not attr.startswith("_")
        ]:
            if type(getattr(self, attr)) in [int, float, str, bool, type(None)]:
                result[attr] = getattr(self, attr)
            elif isinstance(getattr(self, attr), SerializeMixin):
                result[attr] = getattr(self, attr).serialize()
            elif isinstance(getattr(self, attr), list):
                result[attr] = [
                    x.serialize() if isinstance(x, SerializeMixin) else x
                    for x in getattr(self, attr)
                ]
            elif isinstance(getattr(self, attr), dict):
                result[attr] = {
                    x.serialize() if isinstance(x, SerializeMixin) else x
                    for x in getattr(self, attr)
                }
            elif isinstance(getattr(self, attr), tuple) or isinstance(
                getattr(self, attr), set
            ):
                # tuples are immutable, so we have to convert them to lists
                result[attr] = [
                    x.serialize() if isinstance(x, SerializeMixin) else x
                    for x in getattr(self, attr)
                ]

        return result

    def serialize(self):
        """Serialize the object into a dictionary."""
        # get all attributes of the class and its children
        params = self._serialize_object(self.__class__)
        params.update(self._serialize_object(self))

        return {
            "name": serialize_module_name(self),
            "params": params,
        }
