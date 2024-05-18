import enum
from types import NoneType
from typing import Any, Union, TypeVar, Generic, Optional, List, Dict

from pydantic import BaseModel, computed_field


VariableInterfaceType = TypeVar("VariableInterfaceType")


class VariableTypes(enum.Enum):
    String = "string"
    Integer = "integer"
    Float = "float"
    Bool = "bool"


class BaseVariable(BaseModel, Generic[VariableInterfaceType]):
    """
    Represents a single variable. It can be a string, int, float or bool. The type of the variable is determined by the
    type attribute.
    """

    name: str
    value: Union[str, int, float, bool]

    def is_valid(self):
        return True

    def is_valid_value(self, value):
        return True

    @computed_field
    @property
    def type(self) -> str:
        return self.TYPE


class StringVariable(
    BaseVariable[VariableInterfaceType], Generic[VariableInterfaceType]
):
    """
    Represents a single string variable.
    """

    value: str
    name: str

    TYPE: str = VariableTypes.String.value

    def is_valid(self):
        return isinstance(self.value, str)

    def is_valid_value(self, value):
        return isinstance(value, str)


class IntegerVariable(
    BaseVariable[VariableInterfaceType], Generic[VariableInterfaceType]
):
    """
    Represents a single int variable.
    """

    value: int
    name: str

    TYPE: str = VariableTypes.Integer.value

    def is_valid(self):
        return isinstance(self.value, int)

    def is_valid_value(self, value):
        return isinstance(value, int)


class FloatVariable(
    BaseVariable[VariableInterfaceType], Generic[VariableInterfaceType]
):
    """
    Represents a single float variable.
    """

    value: float
    name: str

    TYPE: str = VariableTypes.Float.value

    def is_valid(self):
        return isinstance(self.value, float)

    def is_valid_value(self, value):
        return isinstance(value, float)


class BoolVariable(BaseVariable[VariableInterfaceType], Generic[VariableInterfaceType]):
    """
    Represents a single bool variable.
    """

    value: bool
    name: str

    TYPE: str = VariableTypes.Bool.value

    def is_valid(self):
        return isinstance(self.value, bool)

    def is_valid_value(self, value):
        return isinstance(value, bool)


class VariableReference(BaseModel, Generic[VariableInterfaceType]):
    """
    Represents a reference to a variable.
    """

    name: str

    @computed_field
    @property
    def type(self) -> str:
        return "reference"


VARIABLE_MAPPING = {
    VariableTypes.String.value: StringVariable,
    VariableTypes.Integer.value: IntegerVariable,
    VariableTypes.Float.value: FloatVariable,
    VariableTypes.Bool.value: BoolVariable,
}

BoolParameter = Union[bool, VariableReference]
IntegerParameter = Union[int, VariableReference]
FloatParameter = Union[float, VariableReference]
StringParameter = Union[str, VariableReference]
CausyParameter = Union[BoolParameter, IntegerParameter, FloatParameter, StringParameter]


def validate_variable_values(algorithm, variable_values: Dict[str, Any]):
    """
    Validate the variable values for the algorithm.
    :param algorithm:
    :param variable_values:
    :return:
    """
    algorithm_variables = {avar.name: avar for avar in algorithm.variables}

    for variable_name, variable_value in variable_values.items():
        if variable_name not in algorithm_variables.keys():
            raise ValueError(
                f"Variable {variable_name} not found in the algorithm variables."
            )
        if not algorithm_variables[variable_name].is_valid_value(variable_value):
            raise ValueError(
                f"Variable {variable_name} is not valid."
                f" (should be {algorithm_variables[variable_name].type} but is {type(variable_value)})"
            )

    return True


def resolve_variables(
    variables: List[BaseVariable], variable_values: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Resolve the variables from the list of variables and the variable values coming from the user.
    :param variables:
    :param variable_values:
    :return:
    """
    resolved_variables = {}
    for variable in variables:
        if variable.name in variable_values:
            resolved_variables[variable.name] = variable_values[variable.name]
        else:
            resolved_variables[variable.name] = variable.value

    return resolved_variables


def resolve_variables_to_algorithm(pipeline_steps, variables):
    """
    Resolve the variables to the algorithm.
    :param pipeline_steps:
    :param variables:
    :return:
    """
    for pipeline_step in pipeline_steps:
        for attribute, value in pipeline_step.__dict__.items():
            if isinstance(value, VariableReference):
                if value.name in variables:
                    setattr(pipeline_step, attribute, variables[value.name])
                else:
                    raise ValueError(
                        f'Variable "{value.name}" not found in the variables (used in "{pipeline_step.name}").'
                    )
        # handle cases when we have sub-pipelines like in Loops
        if hasattr(pipeline_step, "pipeline_steps"):
            pipeline_step.pipeline_steps = resolve_variables_to_algorithm(
                pipeline_step.pipeline_steps, variables
            )

    return pipeline_steps


def deserialize_variable(variable_dict: Dict[str, Any]) -> BaseVariable:
    """
    Deserialize the variable from the dictionary.
    :param variable_dict:
    :return:
    """
    if "type" not in variable_dict:
        raise ValueError("Variable type not found.")
    if variable_dict["type"] not in VARIABLE_MAPPING:
        raise ValueError(f"Variable type {variable_dict['type']} not found.")

    return VARIABLE_MAPPING[variable_dict["type"]](**variable_dict)


def deserialize_variable_references(element: object) -> object:
    """
    Deserialize the variable references from the pipeline step.
    :param pipeline_step:
    :return:
    """
    for attribute, value in element.__dict__.items():
        if isinstance(value, dict) and "type" in value and value["type"] == "reference":
            setattr(element, attribute, VariableReference(name=value["name"]))

        if hasattr(value, "__dict__"):
            setattr(element, attribute, deserialize_variable_references(value))

    if hasattr(element, "pipeline_steps"):
        element.pipeline_steps = [
            deserialize_variable_references(pipeline_step)
            for pipeline_step in element.pipeline_steps
        ]

    return element
