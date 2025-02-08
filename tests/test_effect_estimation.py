from causy.causal_discovery.constraint.algorithms.pc import (
    PC_ORIENTATION_RULES,
    PC_EDGE_TYPES,
    PC_GRAPH_UI_EXTENSION,
    PC_DEFAULT_THRESHOLD,
)
from causy.causal_discovery.constraint.independence_tests.common import (
    CorrelationCoefficientTest,
    PartialCorrelationTest,
    ExtendedPartialCorrelationTestMatrix,
)
from causy.causal_discovery.constraint.orientation_rules.pc import ColliderTest
from causy.causal_effect_estimation.multivariate_regression import (
    ComputeDirectEffectsInDAGsMultivariateRegression,
)
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.graph_model import graph_model_factory
from causy.models import Algorithm
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from causy.variables import VariableReference, FloatVariable
from tests.utils import CausyTestCase


class EffectEstimationTestCase(CausyTestCase):
    SEED = 1

    def test_direct_effect_estimation_trivial_case(self):
        PC = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(
                        display_name="Calculate Pearson Correlations"
                    ),
                    CorrelationCoefficientTest(
                        threshold=VariableReference(name="threshold"),
                        display_name="Correlation Coefficient Test",
                    ),
                    PartialCorrelationTest(
                        threshold=VariableReference(name="threshold"),
                        display_name="Partial Correlation Test",
                    ),
                    ExtendedPartialCorrelationTestMatrix(
                        threshold=VariableReference(name="threshold"),
                        display_name="Extended Partial Correlation Test Matrix",
                    ),
                    *PC_ORIENTATION_RULES,
                    ComputeDirectEffectsInDAGsMultivariateRegression(
                        display_name="Compute Direct Effects"
                    ),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[PC_GRAPH_UI_EXTENSION],
                name="PC",
                variables=[FloatVariable(name="threshold", value=PC_DEFAULT_THRESHOLD)],
            )
        )

        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Z"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
            ],
        )

        tst = PC()
        sample_size = 100_000
        test_data, _ = model.generate(sample_size)
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertAlmostEqual(
            tst.graph.edge_value(tst.graph.nodes["X"], tst.graph.nodes["Z"])[
                "direct_effect"
            ],
            5.0,
            0,
        )
        self.assertAlmostEqual(
            tst.graph.edge_value(tst.graph.nodes["Y"], tst.graph.nodes["Z"])[
                "direct_effect"
            ],
            6.0,
            0,
        )

    def test_direct_effect_estimation_basic_example(self):
        PC = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(
                        display_name="Calculate Pearson Correlations"
                    ),
                    CorrelationCoefficientTest(
                        threshold=VariableReference(name="threshold"),
                        display_name="Correlation Coefficient Test",
                    ),
                    PartialCorrelationTest(
                        threshold=VariableReference(name="threshold"),
                        display_name="Partial Correlation Test",
                    ),
                    ExtendedPartialCorrelationTestMatrix(
                        threshold=VariableReference(name="threshold"),
                        display_name="Extended Partial Correlation Test Matrix",
                    ),
                    *PC_ORIENTATION_RULES,
                    ComputeDirectEffectsInDAGsMultivariateRegression(
                        display_name="Compute Direct Effects"
                    ),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[PC_GRAPH_UI_EXTENSION],
                name="PC",
                variables=[FloatVariable(name="threshold", value=PC_DEFAULT_THRESHOLD)],
            )
        )

        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Z"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
                SampleEdge(NodeReference("Z"), NodeReference("V"), 3),
                SampleEdge(NodeReference("Z"), NodeReference("W"), 4),
            ],
        )

        tst = PC()
        sample_size = 1000000
        test_data, graph = model.generate(sample_size)
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        self.assertAlmostEqual(
            tst.graph.edge_value(tst.graph.nodes["X"], tst.graph.nodes["Z"])[
                "direct_effect"
            ],
            5.0,
            0,
        )
        self.assertAlmostEqual(
            tst.graph.edge_value(tst.graph.nodes["Y"], tst.graph.nodes["Z"])[
                "direct_effect"
            ],
            6.0,
            0,
        )

        self.assertAlmostEqual(
            tst.graph.edge_value(tst.graph.nodes["Z"], tst.graph.nodes["V"])[
                "direct_effect"
            ],
            3.0,
            0,
        )

        self.assertAlmostEqual(
            tst.graph.edge_value(tst.graph.nodes["Z"], tst.graph.nodes["W"])[
                "direct_effect"
            ],
            4.0,
            0,
        )

    def test_direct_effect_estimation_partially_directed(self):
        PC = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(
                        display_name="Calculate Pearson Correlations"
                    ),
                    CorrelationCoefficientTest(
                        threshold=VariableReference(name="threshold"),
                        display_name="Correlation Coefficient Test",
                    ),
                    PartialCorrelationTest(
                        threshold=VariableReference(name="threshold"),
                        display_name="Partial Correlation Test",
                    ),
                    ExtendedPartialCorrelationTestMatrix(
                        threshold=VariableReference(name="threshold"),
                        display_name="Extended Partial Correlation Test Matrix",
                    ),
                    *PC_ORIENTATION_RULES,
                    ComputeDirectEffectsInDAGsMultivariateRegression(
                        display_name="Compute Direct Effects"
                    ),
                ],
                edge_types=PC_EDGE_TYPES,
                extensions=[PC_GRAPH_UI_EXTENSION],
                name="PC",
                variables=[FloatVariable(name="threshold", value=PC_DEFAULT_THRESHOLD)],
            )
        )

        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 1),
                SampleEdge(NodeReference("W"), NodeReference("X"), 1),
                SampleEdge(NodeReference("W"), NodeReference("Y"), 1),
            ],
        )

        tst = PC()
        sample_size = 100000
        test_data, graph = model.generate(sample_size)
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(tst.graph, graph)

        self.assertAlmostEqual(
            tst.graph.edge_value(tst.graph.nodes["X"], tst.graph.nodes["Z"])[
                "direct_effect"
            ],
            1.0,
            0,
        )
        self.assertAlmostEqual(
            tst.graph.edge_value(tst.graph.nodes["Y"], tst.graph.nodes["Z"])[
                "direct_effect"
            ],
            1.0,
            0,
        )

        self.assertNotIn(
            "direct_effect",
            tst.graph.edge_value(tst.graph.nodes["W"], tst.graph.nodes["X"]),
        )
        self.assertNotIn(
            "direct_effect",
            tst.graph.edge_value(tst.graph.nodes["W"], tst.graph.nodes["Y"]),
        )
