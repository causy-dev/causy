from causy.exit_conditions import ExitOnNoActions
from causy.generators import (
    PairsWithNeighboursGenerator,
    RandomSampleGenerator,
    BatchGenerator,
    ComparisonSettingsMinMaxGenerator,
)
from causy.graph import graph_model_factory, Loop
from causy.independence_tests import (
    CalculateCorrelations,
    CorrelationCoefficientTest,
    PartialCorrelationTest,
    ExtendedPartialCorrelationTestMatrix,
    ExtendedPartialCorrelationTestMatrixWithTorchBatching,
)
from causy.interfaces import AS_MANY_AS_FIELDS, ComparisonSettings
from causy.orientation_tests import (
    ColliderTest,
    NonColliderTest,
    FurtherOrientTripleTest,
    OrientQuadrupleTest,
    FurtherOrientQuadrupleTest,
)

PC = graph_model_factory(
    pipeline_steps=[
        CalculateCorrelations(),
        CorrelationCoefficientTest(threshold=0.01),
        PartialCorrelationTest(threshold=0.01),
        ExtendedPartialCorrelationTestMatrix(
            threshold=0.01,
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

ParallelPC = graph_model_factory(
    pipeline_steps=[
        CalculateCorrelations(),
        CorrelationCoefficientTest(threshold=0.01),
        PartialCorrelationTest(
            threshold=0.01, parallel=True, chunk_size_parallel_processing=50000
        ),
        ExtendedPartialCorrelationTestMatrix(
            # run first a sampled version of the test so we can minimize the number of tests in the full version
            threshold=0.01,
            chunk_size_parallel_processing=5000,
            parallel=True,
            generator=RandomSampleGenerator(
                comparison_settings=ComparisonSettings(min=4, max=AS_MANY_AS_FIELDS),
                generator=PairsWithNeighboursGenerator(
                    chunked=False, shuffle_combinations=True, comparison_settings=None
                ),
                chunked=False,
                every_nth=200,
            ),
        ),
        ExtendedPartialCorrelationTestMatrix(
            threshold=0.01,
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

ParallelPCTest = graph_model_factory(
    pipeline_steps=[
        CalculateCorrelations(),
        CorrelationCoefficientTest(threshold=0.01),
        PartialCorrelationTest(
            threshold=0.01, parallel=True, chunk_size_parallel_processing=50000
        ),
        ExtendedPartialCorrelationTestMatrixWithTorchBatching(
            threshold=0.01,
            parallel=False,
            generator=ComparisonSettingsMinMaxGenerator(
                comparison_settings=ComparisonSettings(min=4, max=AS_MANY_AS_FIELDS),
                generator=BatchGenerator(
                    generator=RandomSampleGenerator(
                        generator=PairsWithNeighboursGenerator(
                            chunked=False,
                            shuffle_combinations=True,
                            comparison_settings=None,
                        ),
                        chunked=False,
                        every_nth=200,
                    ),
                    chunk_size=6000,
                ),
            ),
        ),
        ExtendedPartialCorrelationTestMatrix(
            threshold=0.01,
            chunk_size_parallel_processing=20000,
            parallel=True,
            generator=PairsWithNeighboursGenerator(
                chunked=False,
                shuffle_combinations=False,
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
