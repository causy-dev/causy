from causy.graph_model import graph_model_factory
from causy.models import Algorithm

PIPELINE = graph_model_factory(
    Algorithm(
        pipeline_steps=[
        ],
        edge_types=[],
        extensions=[],
        variables=[],
        name="{{pipeline_name}}",
    )
)
