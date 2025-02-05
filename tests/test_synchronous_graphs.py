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
from causy.causal_effect_estimation.multivariate_regression import (
    ComputeDirectEffectsInDAGsMultivariateRegression,
)
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.graph_model import graph_model_factory
from causy.models import Algorithm
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from causy.variables import VariableReference, FloatVariable
from tests.utils import CausyTestCase


class PCTestTestCase(CausyTestCase):
    SEED = 1

    def _sample_generator(self):
        rdnv = self.seeded_random.normalvariate
        return IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("X"), NodeReference("Z"), 8),
                SampleEdge(NodeReference("X"), NodeReference("W"), 4),
            ],
            random=lambda: rdnv(0, 1),
        )

    SYNCHRONOUS_PC = graph_model_factory(
        Algorithm(
            pipeline_steps=[
                CalculatePearsonCorrelations(
                    display_name="Calculate Pearson Correlations"
                ),
                CorrelationCoefficientTest(
                    threshold=VariableReference(name="threshold"),
                    display_name="Correlation Coefficient Test",
                    apply_synchronous=True,
                ),
                PartialCorrelationTest(
                    threshold=VariableReference(name="threshold"),
                    display_name="Partial Correlation Test",
                    apply_synchronous=True,
                ),
                ExtendedPartialCorrelationTestMatrix(
                    threshold=VariableReference(name="threshold"),
                    display_name="Extended Partial Correlation Test Matrix",
                    apply_synchronous=True,
                ),
                *PC_ORIENTATION_RULES,
                ComputeDirectEffectsInDAGsMultivariateRegression(
                    display_name="Compute Direct Effects in DAGs Multivariate Regression"
                ),
            ],
            edge_types=PC_EDGE_TYPES,
            extensions=[PC_GRAPH_UI_EXTENSION],
            name="PC",
            variables=[FloatVariable(name="threshold", value=PC_DEFAULT_THRESHOLD)],
        )
    )

    def test_execute_pipeline(self):
        model = self._sample_generator()
        data, graph = model.generate(100)

        pc = self.SYNCHRONOUS_PC()
        pc.create_graph_from_data(data)
        pc.create_graph_from_data(data)
        pc.create_all_possible_edges()
        pc.execute_pipeline_steps()

        self.assertGraphStructureIsEqual(pc.graph, graph)
