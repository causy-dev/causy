from causy.causal_discovery.constraint.algorithms.pc import ColliderTest
from causy.causal_discovery.constraint.independence_tests.common import CorrelationCoefficientTest, \
    PartialCorrelationTest, ExtendedPartialCorrelationTestMatrix
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.contrib.graph_ui import GraphUIExtension
from causy.edge_types import DirectedEdgeUIConfig, UndirectedEdgeUIConfig, DirectedEdge, UndirectedEdge
from causy.graph_model import graph_model_factory
from causy.models import Algorithm
from causy.variables import VariableReference, FloatVariable


CI_DEFAULT_THRESHOLD = 0.005

# wip
CI_ORIENTATION_RULES = [
]

# we need new edge type UI configurations for CI
CI_GRAPH_UI_EXTENSION = GraphUIExtension(
    edges=[
        DirectedEdgeUIConfig(),
        UndirectedEdgeUIConfig(),
    ]
)
# wip
CI_EDGE_TYPES = [DirectedEdge(), UndirectedEdge()]

CI = graph_model_factory(
    Algorithm(
        pipeline_steps=[
            CalculatePearsonCorrelations(display_name="Calculate Pearson Correlations"),
            CorrelationCoefficientTest(
                threshold=VariableReference(name="threshold"),
                display_name="Correlation Coefficient Test",
            ),
            # wip: we need to input another generator in the partial correlation test – pairs of nodes given any subsets of other nodes in no particular order
            PartialCorrelationTest(
                threshold=VariableReference(name="threshold"),
                display_name="Partial Correlation Test",
            ),
            # wip: we need to input another generator in the partial correlation test – pairs of nodes given any subsets of other nodes in no particular order
            ExtendedPartialCorrelationTestMatrix(
                threshold=VariableReference(name="threshold"),
                display_name="Extended Partial Correlation Test Matrix",
            ),
            ColliderTest(),
            *CI_ORIENTATION_RULES,
        ],
        edge_types=CI_EDGE_TYPES,
        extensions=[CI_GRAPH_UI_EXTENSION],
        name="CI",
        variables=[FloatVariable(name="threshold", value=CI_DEFAULT_THRESHOLD)],
    )
)
