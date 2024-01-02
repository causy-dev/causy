from causy.common_pipeline_steps.exit_conditions import ExitOnNoActions
from causy.generators import PairsWithNeighboursGenerator, RandomSampleGenerator
from causy.graph_model import graph_model_factory
from causy.common_pipeline_steps.logic import Loop
from causy.independence_tests.common import (
    CorrelationCoefficientTest,
    PartialCorrelationTest,
    ExtendedPartialCorrelationTestMatrix,
)
from causy.common_pipeline_steps.calculation import (
    CalculatePearsonCorrelations,
)
from causy.interfaces import AS_MANY_AS_FIELDS, ComparisonSettings
from causy.orientation_rules.pc import (
    ColliderTest,
    NonColliderTest,
    FurtherOrientTripleTest,
    OrientQuadrupleTest,
    FurtherOrientQuadrupleTest,
)

PC = graph_model_factory(
    pipeline_steps=[
        CalculatePearsonCorrelations(),
        CorrelationCoefficientTest(threshold=0.005),
        PartialCorrelationTest(threshold=0.005),
        ExtendedPartialCorrelationTestMatrix(threshold=0.005),
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
)


ParallelPC = graph_model_factory(
    pipeline_steps=[
        CalculatePearsonCorrelations(),
        CorrelationCoefficientTest(threshold=0.005),
        PartialCorrelationTest(
            threshold=0.005, parallel=True, chunk_size_parallel_processing=50000
        ),
        ExtendedPartialCorrelationTestMatrix(
            # run first a sampled version of the test so we can minimize the number of tests in the full version
            threshold=0.005,
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
            threshold=0.005,
            chunk_size_parallel_processing=20000,
            parallel=True,
            generator=PairsWithNeighboursGenerator(
                chunked=False,
                shuffle_combinations=True,
                comparison_settings=ComparisonSettings(min=4, max=AS_MANY_AS_FIELDS),
            ),
        ),
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
)
