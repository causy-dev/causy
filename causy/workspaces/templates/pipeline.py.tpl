from causy.graph_model import graph_model_factory
from causy.interfaces import CausyAlgorithm

PIPELINE = graph_model_factory(
    CausyAlgorithm(
        pipeline_steps=[
        ],
        edge_types=[],
        extensions=[],
        name="{{pipeline_name}}",
    )
)
