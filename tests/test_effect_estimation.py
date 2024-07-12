from causy.causal_discovery.constraint.algorithms.pc import PC_ORIENTATION_RULES, PC_EDGE_TYPES, PC_GRAPH_UI_EXTENSION, \
    PC_DEFAULT_THRESHOLD
from causy.causal_discovery.constraint.independence_tests.common import CorrelationCoefficientTest, \
    PartialCorrelationTest, ExtendedPartialCorrelationTestMatrix
from causy.causal_effect_estimation.multivariate_regression import ComputeDirectEffectsMultivariateRegression
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.graph_model import graph_model_factory
from causy.models import Algorithm
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from causy.variables import VariableReference, FloatVariable
from tests.utils import CausyTestCase


class EffectEstimationTestCase(CausyTestCase):
    SEED = 1

    def test_direct_effect_estimation(self):
        # In this example, three direct effects are identifiable, only these can be checked
        PC = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(display_name="Calculate Pearson Correlations"),
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
                    ComputeDirectEffectsMultivariateRegression(
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
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Z"), NodeReference("Y"), 6),
                SampleEdge(NodeReference("W"), NodeReference("Y"), 2),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 3),
                SampleEdge(NodeReference("X"), NodeReference("W"), 4),
            ],
        )
        tst = PC()
        sample_size = 100_000
        test_data, _ = model.generate(sample_size)
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()

        self.assertAlmostEqual(tst.graph.edge_value(tst.graph.nodes["X"], tst.graph.nodes["Y"])["direct_effect"], 5, 1)
        self.assertAlmostEqual(tst.graph.edge_value(tst.graph.nodes["Z"], tst.graph.nodes["Y"])["direct_effect"], 6, 1)
        self.assertAlmostEqual(tst.graph.edge_value(tst.graph.nodes["W"], tst.graph.nodes["Y"])["direct_effect"], 2, 1)
    def test_direct_effect_estimation_weird_graph(self):
        """
        Here, the wrong graph is discovered, so the effects are also wrong – check which assumption is violated such that the wrong graph is discovered from toy data
        Leaving this here to think about how to deal with such edge cases.
        """
        PC = graph_model_factory(
            Algorithm(
                pipeline_steps=[
                    CalculatePearsonCorrelations(display_name="Calculate Pearson Correlations"),
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
                    ComputeDirectEffectsMultivariateRegression(
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
                SampleEdge(NodeReference("A"), NodeReference("C"), 1),
                SampleEdge(NodeReference("B"), NodeReference("C"), 2),
                SampleEdge(NodeReference("A"), NodeReference("D"), 3),
                SampleEdge(NodeReference("B"), NodeReference("D"), 1),
                SampleEdge(NodeReference("C"), NodeReference("D"), 1),
                SampleEdge(NodeReference("B"), NodeReference("E"), 4),
                SampleEdge(NodeReference("E"), NodeReference("F"), 5),
                SampleEdge(NodeReference("B"), NodeReference("F"), 6),
                SampleEdge(NodeReference("C"), NodeReference("F"), 1),
                SampleEdge(NodeReference("D"), NodeReference("F"), 1),
            ],
        )
        tst = PC()
        sample_size = 100_000
        test_data, _ = model.generate(sample_size)
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()