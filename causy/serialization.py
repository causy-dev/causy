from causy.graph_utils import serialize_module_name


def serialize_model(model, algorithm_name: str = None):
    """Serialize the model into a dictionary."""
    output = []
    for step in model.pipeline_steps:
        output.append(step.serialize())

    return {"name": algorithm_name, "steps": output}


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
