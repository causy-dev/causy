import copy
from unittest import skip

from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.common_pipeline_steps.placeholder import PlaceholderTest
from causy.graph_model import graph_model_factory
from causy.graph_utils import (
    serialize_module_name,
    load_pipeline_artefact_by_definition,
    load_pipeline_steps_by_definition,
)
from causy.models import CausyAlgorithm
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from causy.variables import (
    StringVariable,
    FloatVariable,
    IntegerVariable,
    BoolVariable,
    validate_variable_values,
    VariableReference,
    resolve_variables,
    resolve_variables_to_algorithm_for_pipeline_steps,
)

from tests.utils import CausyTestCase


class VariablesTestCase(CausyTestCase):
    def test_validate_str_variable(self):
        variable = StringVariable(name="threshold", value="default")
        self.assertEqual(variable.is_valid_value("test"), True)
        self.assertEqual(variable.is_valid_value(1), False)
        self.assertEqual(variable.is_valid_value(1.0), False)
        self.assertEqual(variable.is_valid_value(True), False)

        with self.assertRaises(ValueError):
            variable.validate_value(1)

        with self.assertRaises(ValueError):
            variable.validate_value(1.0)

        with self.assertRaises(ValueError):
            variable.validate_value(True)

        with self.assertRaises(ValueError):
            variable.validate_value(None)

        with self.assertRaises(ValueError):
            variable.validate_value([])

        variable.validate_value("test")
        variable.validate_value("default")

        self.assertEqual(variable.is_valid(), True)

        variable = StringVariable(
            name="threshold", value="default", choices=["test", "default"]
        )
        self.assertEqual(variable.is_valid_value("test"), True)
        self.assertEqual(variable.is_valid_value("default"), True)
        self.assertEqual(variable.is_valid_value("test1"), False)
        with self.assertRaises(ValueError):
            variable.validate_value("test1")

    def test_validate_float_variable(self):
        variable = FloatVariable(name="threshold", value=0.5)
        self.assertEqual(variable.is_valid_value(1.0), True)
        self.assertEqual(variable.is_valid_value(1), False)
        self.assertEqual(variable.is_valid_value(0.5), True)
        self.assertEqual(variable.is_valid_value(True), False)

        with self.assertRaises(ValueError):
            variable.validate_value("test")

        with self.assertRaises(ValueError):
            variable.validate_value(True)

        with self.assertRaises(ValueError):
            variable.validate_value(None)

        with self.assertRaises(ValueError):
            variable.validate_value([])

        with self.assertRaises(ValueError):
            variable.validate_value(1)

        self.assertEqual(variable.is_valid(), True)

        variable.validate_value(0.5)
        variable.validate_value(1.0)

        variable = FloatVariable(name="threshold", value=0.5, choices=[0.5, 1.0])
        self.assertEqual(variable.is_valid_value(0.5), True)
        self.assertEqual(variable.is_valid_value(1.0), True)
        self.assertEqual(variable.is_valid_value(0.6), False)
        with self.assertRaises(ValueError):
            variable.validate_value(0.6)

    def test_validate_int_variable(self):
        variable = IntegerVariable(name="threshold", value=1)
        self.assertEqual(variable.is_valid_value(1), True)
        self.assertEqual(variable.is_valid_value(1.0), False)
        self.assertEqual(variable.is_valid_value(True), False)

        with self.assertRaises(ValueError):
            variable.validate_value("test")

        with self.assertRaises(ValueError):
            variable.validate_value(1.0)

        with self.assertRaises(ValueError):
            variable.validate_value(True)

        with self.assertRaises(ValueError):
            variable.validate_value(None)

        with self.assertRaises(ValueError):
            variable.validate_value([])

        variable.validate_value(1)
        variable.validate_value(0)
        variable.validate_value(100)
        variable.validate_value(-100)
        self.assertEqual(variable.is_valid(), True)

        variable = IntegerVariable(name="threshold", value=1, choices=[0, 1, 100])
        self.assertEqual(variable.is_valid_value(0), True)
        self.assertEqual(variable.is_valid_value(1), True)
        self.assertEqual(variable.is_valid_value(100), True)
        self.assertEqual(variable.is_valid_value(101), False)
        with self.assertRaises(ValueError):
            variable.validate_value(101)

    def test_validate_bool_variable(self):
        variable = BoolVariable(name="threshold", value=True)
        self.assertEqual(variable.is_valid_value(True), True)
        self.assertEqual(variable.is_valid_value(False), True)
        self.assertEqual(variable.is_valid_value(1), False)
        self.assertEqual(variable.is_valid_value(1.0), False)

        with self.assertRaises(ValueError):
            variable.validate_value("test")

        with self.assertRaises(ValueError):
            variable.validate_value(1.0)

        with self.assertRaises(ValueError):
            variable.validate_value(None)

        with self.assertRaises(ValueError):
            variable.validate_value([])

        variable.validate_value(True)
        variable.validate_value(False)
        self.assertEqual(variable.is_valid(), True)

        variable = BoolVariable(name="threshold", value=True, choices=[True])
        self.assertEqual(variable.is_valid_value(True), True)
        self.assertEqual(variable.is_valid_value(1), False)
        self.assertEqual(variable.is_valid_value(0), False)
        self.assertEqual(variable.is_valid_value("True"), False)
        self.assertEqual(variable.is_valid_value("False"), False)

        with self.assertRaises(ValueError):
            variable.validate_value("True")

        with self.assertRaises(ValueError):
            variable.validate_value(False)

    def test_validate_variable_values(self):
        algorithm = graph_model_factory(
            CausyAlgorithm(
                pipeline_steps=[],
                edge_types=[],
                extensions=[],
                name="Test variable validation",
                variables=[
                    StringVariable(name="a_string", value="default"),
                    IntegerVariable(name="an_int", value=1),
                    BoolVariable(name="a_bool", value=True),
                    FloatVariable(name="a_float", value=0.1),
                ],
            )
        )()

        with self.assertRaises(ValueError):
            validate_variable_values(algorithm.algorithm, {"a_string": 1})

        with self.assertRaises(ValueError):
            validate_variable_values(algorithm.algorithm, {"an_int": "test"})

        with self.assertRaises(ValueError):
            validate_variable_values(algorithm.algorithm, {"a_bool": 1})

        with self.assertRaises(ValueError):
            validate_variable_values(algorithm.algorithm, {"a_float": "test"})

        with self.assertRaises(ValueError):
            validate_variable_values(algorithm.algorithm, {"a_float": True})

        validate_variable_values(algorithm.algorithm, {"a_string": "test"})
        validate_variable_values(algorithm.algorithm, {"an_int": 2})
        validate_variable_values(algorithm.algorithm, {"a_bool": False})
        validate_variable_values(algorithm.algorithm, {"a_float": 0.2})

        with self.assertRaises(ValueError):
            validate_variable_values(algorithm.algorithm, {"another_var": 0.21})

    def test_resolve_variables(self):
        algorithm = graph_model_factory(
            CausyAlgorithm(
                pipeline_steps=[
                    PlaceholderTest(
                        placeholder_str=VariableReference(name="a_string"),
                        placeholder_int=VariableReference(name="an_int"),
                        placeholder_float=VariableReference(name="a_float"),
                        placeholder_bool=VariableReference(name="a_bool"),
                    )
                ],
                edge_types=[],
                extensions=[],
                name="Test variable resolution",
                variables=[
                    StringVariable(name="a_string", value="default"),
                    IntegerVariable(name="an_int", value=1),
                    BoolVariable(name="a_bool", value=True),
                    FloatVariable(name="a_float", value=0.1),
                ],
            )
        )()

        resolved_variables = resolve_variables(
            algorithm._original_algorithm.variables,
            {"a_string": "test", "an_int": 2, "a_bool": False, "a_float": 0.2},
        )

        self.assertEqual(
            resolved_variables,
            {"a_string": "test", "an_int": 2, "a_bool": False, "a_float": 0.2},
        )

        resolved_variables = resolve_variables(
            algorithm._original_algorithm.variables,
            {
                "a_string": "test",
            },
        )

        self.assertEqual(
            resolved_variables,
            {"a_string": "test", "an_int": 1, "a_bool": True, "a_float": 0.1},
        )

    def test_resolve_variables_to_algorithm(self):
        algorithm = graph_model_factory(
            CausyAlgorithm(
                pipeline_steps=[
                    PlaceholderTest(
                        placeholder_str=VariableReference(name="a_string"),
                        placeholder_int=VariableReference(name="an_int"),
                        placeholder_float=VariableReference(name="a_float"),
                        placeholder_bool=VariableReference(name="a_bool"),
                    )
                ],
                edge_types=[],
                extensions=[],
                name="Test variable resolution",
                variables=[
                    StringVariable(name="a_string", value="default"),
                    IntegerVariable(name="an_int", value=1),
                    BoolVariable(name="a_bool", value=True),
                    FloatVariable(name="a_float", value=0.1),
                ],
            )
        )

        resolved_variables = resolve_variables_to_algorithm_for_pipeline_steps(
            algorithm._original_algorithm.pipeline_steps,
            {"a_string": "test", "an_int": 2, "a_bool": False, "a_float": 0.2},
        )

        algorithm = graph_model_factory(
            CausyAlgorithm(
                pipeline_steps=[
                    PlaceholderTest(
                        placeholder_str=VariableReference(name="a_string"),
                        placeholder_int=VariableReference(name="an_int"),
                        placeholder_float=VariableReference(name="a_float"),
                        placeholder_bool=VariableReference(name="a_bool"),
                    )
                ],
                edge_types=[],
                extensions=[],
                name="Test variable resolution",
                variables=[
                    StringVariable(name="a_string", value="default"),
                    IntegerVariable(name="an_int", value=1),
                    BoolVariable(name="a_bool", value=True),
                    FloatVariable(name="a_float", value=0.1),
                ],
            )
        )
        with self.assertRaises(ValueError):
            resolved_variables = resolve_variables_to_algorithm_for_pipeline_steps(
                algorithm._original_algorithm.pipeline_steps,
                {
                    "a_string": "test",
                },
            )

        algorithm = CausyAlgorithm(
            pipeline_steps=[
                PlaceholderTest(
                    placeholder_str=VariableReference(name="not_defined"),
                )
            ],
            edge_types=[],
            extensions=[],
            name="Test variable resolution",
            variables=[],
        )

        resolved_variables = resolve_variables_to_algorithm_for_pipeline_steps(
            algorithm.pipeline_steps,
            {
                "not_defined": "test",
            },
        )

        algorithm = CausyAlgorithm(
            pipeline_steps=[
                PlaceholderTest(
                    placeholder_str=VariableReference(name="not_defined"),
                )
            ],
            edge_types=[],
            extensions=[],
            name="Test variable resolution",
            variables=[],
        )
        with self.assertRaises(ValueError):
            resolved_variables = resolve_variables_to_algorithm_for_pipeline_steps(
                algorithm.pipeline_steps, {}
            )
