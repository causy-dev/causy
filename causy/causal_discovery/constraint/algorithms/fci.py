from typing import Union

from causy.causal_discovery.constraint.independence_tests.common import CorrelationCoefficientTest, \
    PartialCorrelationTest, ExtendedPartialCorrelationTestMatrix
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.contrib.graph_ui import GraphUIExtension
from causy.edge_types import DirectedEdgeUIConfig, UndirectedEdgeUIConfig, DirectedEdge, UndirectedEdge
from causy.graph import Node
from causy.graph_model import graph_model_factory
from causy.interfaces import ExtensionInterface
from causy.models import Algorithm
from causy.variables import VariableReference, FloatVariable


class InducingPathExtension(ExtensionInterface):
    class GraphAccessMixin:
        def inducing_path_exists(
            self, u: Union[Node, str], v: Union[Node, str]
        ) -> bool:
            """
            Check if an inducing path from u to v exists.
            An inducing path from u to v is a directed reference from u to v on which all mediators are colliders.
            :param u: node u
            :param v: node v
            :return: True if an inducing path exists, False otherwise
            """

            if isinstance(u, Node):
                u = u.id
            if isinstance(v, Node):
                v = v.id

            if not self.directed_path_exists(u, v):
                return False
            for path in self.directed_paths(u, v):
                for i in range(1, len(path) - 1):
                    r, w = path[i]
                    if not self.bidirected_edge_exists(r, w):
                        # TODO: check if this is correct (@sof)
                        return True
            return False

FCI_DEFAULT_THRESHOLD = 0.005

# wip
FCI_ORIENTATION_RULES = [
]

# we need new edge type UI configurations for FCI
FCI_GRAPH_UI_EXTENSION = GraphUIExtension(
    edges=[
        DirectedEdgeUIConfig(),
        UndirectedEdgeUIConfig(),
    ]
)

# wip
FCI_EDGE_TYPES = [DirectedEdge(), UndirectedEdge()]


class CollidersTest:
    pass

# wip
FCI = graph_model_factory(
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
            CollidersTest(),
            *FCI_ORIENTATION_RULES,
        ],
        edge_types=FCI_EDGE_TYPES,
        extensions=[FCI_GRAPH_UI_EXTENSION],
        name="FCI",
        variables=[FloatVariable(name="threshold", value=FCI_DEFAULT_THRESHOLD)],
    )
)
