from causy.causal_effect_estimation.multivariate_regression import (
    ComputeDirectEffectsMultivariateRegression,
)
from causy.common_pipeline_steps.exit_conditions import ExitOnNoActions
from causy.contrib.graph_ui import GraphUIExtension
from causy.edge_types import (
    DirectedEdge,
    UndirectedEdge,
    DirectedEdgeUIConfig,
    UndirectedEdgeUIConfig,
)
from causy.generators import PairsWithNeighboursGenerator, RandomSampleGenerator
from causy.graph_model import graph_model_factory
from causy.common_pipeline_steps.logic import Loop, ApplyActionsTogether
from causy.causal_discovery.constraint.independence_tests.common import (
    CorrelationCoefficientTest,
    PartialCorrelationTest,
    ExtendedPartialCorrelationTestMatrix,
)
from causy.common_pipeline_steps.calculation import (
    CalculatePearsonCorrelations,
)
from causy.interfaces import AS_MANY_AS_FIELDS
from causy.models import ComparisonSettings, Algorithm
from causy.causal_discovery.constraint.orientation_rules.pc import (
    ColliderTest,
    NonColliderTest,
    FurtherOrientTripleTest,
    OrientQuadrupleTest,
    FurtherOrientQuadrupleTest,
)
from causy.variables import (
    FloatVariable,
    VariableReference,
    IntegerVariable,
)

PC_DEFAULT_THRESHOLD = 0.005

PC_ORIENTATION_RULES = [
    ColliderTest(display_name="Collider Test"),
    Loop(
        pipeline_steps=[
            NonColliderTest(display_name="Non-Collider Test"),
            FurtherOrientTripleTest(display_name="Further Orient Triple Test"),
            OrientQuadrupleTest(display_name="Orient Quadruple Test"),
            FurtherOrientQuadrupleTest(display_name="Further Orient Quadruple Test"),
        ],
        display_name="Orientation Rules Loop",
        exit_condition=ExitOnNoActions(),
    ),
]

PC_GRAPH_UI_EXTENSION = GraphUIExtension(
    edges=[
        DirectedEdgeUIConfig(),
        UndirectedEdgeUIConfig(),
    ]
)

PC_EDGE_TYPES = [DirectedEdge(), UndirectedEdge()]

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

PCStable = graph_model_factory(
    Algorithm(
        pipeline_steps=[
            CalculatePearsonCorrelations(),
            ApplyActionsTogether(
                pipeline_steps=[
                    CorrelationCoefficientTest(
                        threshold=VariableReference(name="threshold")
                    ),
                    PartialCorrelationTest(
                        threshold=VariableReference(name="threshold"),
                    ),
                    ExtendedPartialCorrelationTestMatrix(
                        threshold=VariableReference(name="threshold"),
                    ),
                ]
            ),
            *PC_ORIENTATION_RULES,
            ComputeDirectEffectsMultivariateRegression(),
        ],
        edge_types=PC_EDGE_TYPES,
        extensions=[PC_GRAPH_UI_EXTENSION],
        name="PCStable",
        variables=[FloatVariable(name="threshold", value=PC_DEFAULT_THRESHOLD)],
    )
)


ParallelPC = graph_model_factory(
    Algorithm(
        pipeline_steps=[
            CalculatePearsonCorrelations(display_name="Calculate Pearson Correlations"),
            CorrelationCoefficientTest(
                threshold=VariableReference(name="threshold"),
                display_name="Correlation Coefficient Test",
            ),
            PartialCorrelationTest(
                threshold=VariableReference(name="threshold"),
                parallel=True,
                chunk_size_parallel_processing=50000,
                display_name="Partial Correlation Test",
            ),
            ExtendedPartialCorrelationTestMatrix(
                # run first a sampled version of the test so we can minimize the number of tests in the full version
                threshold=VariableReference(name="threshold"),
                display_name="Sampled Extended Partial Correlation Test Matrix",
                chunk_size_parallel_processing=5000,
                parallel=True,
                generator=RandomSampleGenerator(
                    generator=PairsWithNeighboursGenerator(
                        chunked=False,
                        shuffle_combinations=True,
                        comparison_settings=ComparisonSettings(
                            min=4, max=AS_MANY_AS_FIELDS
                        ),
                    ),
                    chunked=False,
                    every_nth=VariableReference(name="every_nth_sample"),
                ),
            ),
            ExtendedPartialCorrelationTestMatrix(
                threshold=VariableReference(name="threshold"),
                display_name="Extended Partial Correlation Test Matrix",
                chunk_size_parallel_processing=20000,
                parallel=True,
                generator=PairsWithNeighboursGenerator(
                    chunked=False,
                    shuffle_combinations=True,
                    comparison_settings=ComparisonSettings(
                        min=4, max=AS_MANY_AS_FIELDS
                    ),
                ),
            ),
            *PC_ORIENTATION_RULES,
            ComputeDirectEffectsMultivariateRegression(
                display_name="Compute Direct Effects"
            ),
        ],
        edge_types=PC_EDGE_TYPES,
        extensions=[PC_GRAPH_UI_EXTENSION],
        name="ParallelPC",
        variables=[
            FloatVariable(name="threshold", value=PC_DEFAULT_THRESHOLD),
            IntegerVariable(name="every_nth_sample", value=200),
        ],
    )
)
