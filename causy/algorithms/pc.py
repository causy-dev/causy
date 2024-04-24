from causy.causal_effect_estimation.multivariate_regression import (
    ComputeDirectEffectsMultivariateRegression,
)
from causy.common_pipeline_steps.exit_conditions import ExitOnNoActions
from causy.edge_types import DirectedEdge, UndirectedEdge
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

PC_ORIENTATION_RULES = [
    ColliderTest(),
    Loop(
        pipeline_steps=[
            NonColliderTest(),
            FurtherOrientTripleTest(),
            OrientQuadrupleTest(),
            FurtherOrientQuadrupleTest(),
        ],
        exit_condition=ExitOnNoActions(),
    ),
]

PC_EDGE_TYPES = [DirectedEdge(), UndirectedEdge()]

PC = graph_model_factory(
    CausyAlgorithm(
        pipeline_steps=[
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.05),
            PartialCorrelationTest(threshold=0.05),
            ExtendedPartialCorrelationTestMatrix(threshold=0.05),
            *PC_ORIENTATION_RULES,
            ComputeDirectEffectsMultivariateRegression(),
        ],
        edge_types=PC_EDGE_TYPES,
        name="PC",
    )
)

PCStable = graph_model_factory(
    CausyAlgorithm(
        pipeline_steps=[
            CalculatePearsonCorrelations(),
            ApplyActionsTogether(
                pipeline_steps=[
                    CorrelationCoefficientTest(threshold=0.01),
                    PartialCorrelationTest(threshold=0.01),
                    ExtendedPartialCorrelationTestMatrix(threshold=0.01),
                ]
            ),
            *PC_ORIENTATION_RULES,
            ComputeDirectEffectsMultivariateRegression(),
        ],
        edge_types=PC_EDGE_TYPES,
        name="PCStable",
    )
)


ParallelPC = graph_model_factory(
    CausyAlgorithm(
        pipeline_steps=[
            CalculatePearsonCorrelations(),
            CorrelationCoefficientTest(threshold=0.001),
            PartialCorrelationTest(
                threshold=0.001, parallel=True, chunk_size_parallel_processing=50000
            ),
            ExtendedPartialCorrelationTestMatrix(
                # run first a sampled version of the test so we can minimize the number of tests in the full version
                threshold=0.001,
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
                threshold=0.001,
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
            ComputeDirectEffectsMultivariateRegression(),
        ],
        edge_types=PC_EDGE_TYPES,
        name="ParallelPC",
    )
)
