import copy
import enum
from types import NoneType
from typing import Any, Union, TypeVar, Generic, Optional, List, Dict

from pydantic import BaseModel, computed_field


VariableInterfaceType = TypeVar("VariableInterfaceType")

VariableType = Union[str, int, float, bool]


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

    def __init__(self, **data):
        super().__init__(**data)
        self.validate_value(self.value)

    name: str
    value: Union[str, int, float, bool]
    choices: Optional[List[Union[str, int, float, bool]]] = None

    def is_valid(self):
        return self.is_valid_value(self.value)

    def is_valid_value(self, value):
        try:
            self.validate_value(value)
            return True
        except ValueError:
            return False

    def validate_value(self, value):
        if not isinstance(value, self._PYTHON_TYPE):
            raise ValueError(
                f"Variable {self.name} is not valid."
                f" (should be {self.type} but is {type(value)})"
            )

        if self.choices and value not in self.choices:
            raise ValueError(
                f"Value {value} is not in the list of choices: {self.choices}"
            )

    @computed_field
    @property
    def type(self) -> Optional[str]:
        return self._TYPE


class StringVariable(
    BaseVariable[VariableInterfaceType], Generic[VariableInterfaceType]
):
    """
    Represents a single string variable.
    """

    value: str
    name: str

    _TYPE: Optional[str] = VariableTypes.String.value
    _PYTHON_TYPE: Optional[type] = str


class IntegerVariable(
    BaseVariable[VariableInterfaceType], Generic[VariableInterfaceType]
):
    """
    Represents a single int variable.
    """

    value: int
    name: str

    _TYPE: str = VariableTypes.Integer.value
    _PYTHON_TYPE: Optional[type] = int

    def validate_value(self, value):
        # check if the value is a boolean and raise an error
        # we do this because in python bool is a subclass of int
        if isinstance(value, bool):
            raise ValueError(
                f"Variable {self.name} is not valid."
                f" (should be {self.type} but is {type(value)})"
            )
        super().validate_value(value)


class FloatVariable(
    BaseVariable[VariableInterfaceType], Generic[VariableInterfaceType]
):
    """
    Represents a single float variable.
    """

    value: float
    name: str

    _TYPE: str = VariableTypes.Float.value
    _PYTHON_TYPE: Optional[type] = float


class BoolVariable(BaseVariable[VariableInterfaceType], Generic[VariableInterfaceType]):
    """
    Represents a single bool variable.
    """

    value: bool
    name: str

    _TYPE: str = VariableTypes.Bool.value
    _PYTHON_TYPE: Optional[type] = bool


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


def validate_variable_values(algorithm, variable_values: Dict[str, VariableType]):
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
        algorithm_variables[variable_name].validate_value(variable_value)

    return True


def resolve_variables(
    variables: List[BaseVariable], variable_values: Dict[str, VariableType]
) -> Dict[str, VariableType]:
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


def resolve_variable_to_object(obj: Any, variables):
    """
    Resolve the variables to the object.
    :param obj:
    :param variables:
    :return:
    """
    for attribute, value in obj.__dict__.items():
        if isinstance(value, VariableReference):
            if value.name in variables:
                obj.__dict__[attribute] = variables[value.name]
            else:
                raise ValueError(f'Variable "{value.name}" not found in the variables.')
        elif (
            hasattr(value, "__dict__")
            and not isinstance(value, NoneType)
            and not hasattr(value, "value")
        ):
            # we check for value because we don't want to resolve the variable if it's a variable object itself or a Enum
            obj.__dict__[attribute] = resolve_variable_to_object(value, variables)
    return obj


def resolve_variables_to_algorithm_for_pipeline_steps(pipeline_steps, variables):
    """
    Resolve the variables to the algorithm.
    :param pipeline_steps:
    :param variables:
    :return:
    """
    for k, pipeline_step in enumerate(pipeline_steps):
        pipeline_steps[k] = resolve_variable_to_object(pipeline_step, variables)
        # handle cases when we have sub-pipelines like in Loops
        if hasattr(pipeline_step, "pipeline_steps"):
            pipeline_step.pipeline_steps = (
                resolve_variables_to_algorithm_for_pipeline_steps(
                    pipeline_step.pipeline_steps, variables
                )
            )

    return pipeline_steps


def deserialize_variable(variable_dict: Dict[str, VariableType]) -> BaseVariable:
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
