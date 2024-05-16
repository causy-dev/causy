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
from causy.independence_tests.common import (
    CorrelationCoefficientTest,
    PartialCorrelationTest,
    ExtendedPartialCorrelationTestMatrix,
)
from causy.common_pipeline_steps.calculation import (
    CalculatePearsonCorrelations,
)
from causy.interfaces import AS_MANY_AS_FIELDS, ComparisonSettings, CausyAlgorithm
from causy.orientation_rules.pc import (
    ColliderTest,
    NonColliderTest,
    FurtherOrientTripleTest,
    OrientQuadrupleTest,
    FurtherOrientQuadrupleTest,
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
    CausyAlgorithm(
        pipeline_steps=[
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=PC_DEFAULT_THRESHOLD),
            PartialCorrelationTest(threshold=PC_DEFAULT_THRESHOLD),
            ExtendedPartialCorrelationTestMatrix(threshold=PC_DEFAULT_THRESHOLD),
            *PC_ORIENTATION_RULES,
            ComputeDirectEffectsMultivariateRegression(),
        ],
        edge_types=PC_EDGE_TYPES,
        extensions=[PC_GRAPH_UI_EXTENSION],
        name="PC",
    )
)

PCStable = graph_model_factory(
    CausyAlgorithm(
        pipeline_steps=[
            CalculatePearsonCorrelations(),
            ApplyActionsTogether(
                pipeline_steps=[
                    CorrelationCoefficientTest(threshold=PC_DEFAULT_THRESHOLD),
                    PartialCorrelationTest(threshold=PC_DEFAULT_THRESHOLD),
                    ExtendedPartialCorrelationTestMatrix(
                        threshold=PC_DEFAULT_THRESHOLD
                    ),
                ]
            ),
            *PC_ORIENTATION_RULES,
            ComputeDirectEffectsMultivariateRegression(),
        ],
        edge_types=PC_EDGE_TYPES,
        extensions=[PC_GRAPH_UI_EXTENSION],
        name="PCStable",
    )
)


ParallelPC = graph_model_factory(
    CausyAlgorithm(
        pipeline_steps=[
            CalculatePearsonCorrelations(display_name="Calculate Pearson Correlations"),
            CorrelationCoefficientTest(
                threshold=PC_DEFAULT_THRESHOLD,
                display_name="Correlation Coefficient Test",
            ),
            PartialCorrelationTest(
                threshold=PC_DEFAULT_THRESHOLD,
                parallel=True,
                chunk_size_parallel_processing=50000,
                display_name="Partial Correlation Test",
            ),
            ExtendedPartialCorrelationTestMatrix(
                # run first a sampled version of the test so we can minimize the number of tests in the full version
                threshold=PC_DEFAULT_THRESHOLD,
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
                    every_nth=200,
                ),
            ),
            ExtendedPartialCorrelationTestMatrix(
                threshold=PC_DEFAULT_THRESHOLD,
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
    )
)
