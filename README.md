> [!WARNING]
> causy is currently in a very early stage of development. We do not recommend using it in production.
# causy

Causal discovery made easy.

## Installation
Currently we only support python 3.11. To install causy run
```bash
pip install causy
```

## Usage
Causy can be used via CLI or via code. 

### Usage via CLI

Run causy with one of the default algorithms
```bash
causy execute --help
causy execute your_data.json --algorithm PC --output-file output.json
```

The data you can use is a json file with a list of dictionaries. Each dictionary represents a data point. The keys of the dictionary are the variable names and the values are the values of the variables. The values can be either numeric or categorical. 

```json
[
    {"a": 1, "b": 0.3},
    {"a": 0.5, "b": 0.2}
]
```

You can customize your causy pipeline by ejecting and modifying the pipeline file.
```bash
causy eject PC pc.json
# edit pc.json
causy execute your_data.json --pipeline pc.json
```

This might be useful if you want to use a custom algorithm or if you want to customize the pipeline of a default algorithm.


### Usage via Code

Use a default algorithm

```python
from causy.algorithms import PC
from causy.utils import retrieve_edges

model = PC()
model.create_graph_from_data(
    [
        {"a": 1, "b": 0.3},
        {"a": 0.5, "b": 0.2}
    ]
)
model.create_all_possible_edges()
model.execute_pipeline_steps()
edges = retrieve_edges(model.graph)

for edge in edges:
    print(
        f"{edge[0].name} -> {edge[1].name}: {model.graph.edges[edge[0]][edge[1]]}"
    )

```

Use a custom algorithm

```python
from causy.exit_conditions import ExitOnNoActions
from causy.graph import graph_model_factory, Loop
from causy.independence_tests import (
    CalculateCorrelations,
    CorrelationCoefficientTest,
    PartialCorrelationTest,
    ExtendedPartialCorrelationTestMatrix,
)
from causy.orientation_tests import (
    ColliderTest,
    NonColliderTest,
    FurtherOrientTripleTest,
    OrientQuadrupleTest,
    FurtherOrientQuadrupleTest,
)
from causy.utils import retrieve_edges

CustomPC = graph_model_factory(
    pipeline_steps=[
        CalculateCorrelations(),
        CorrelationCoefficientTest(threshold=0.1),
        PartialCorrelationTest(threshold=0.01),
        ExtendedPartialCorrelationTestMatrix(threshold=0.01),
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

model = CustomPC()

model.create_graph_from_data(
    [
        {"a": 1, "b": 0.3},
        {"a": 0.5, "b": 0.2}
    ]
)
model.create_all_possible_edges()
model.execute_pipeline_steps()
edges = retrieve_edges(model.graph)

for edge in edges:
    print(
        f"{edge[0].name} -> {edge[1].name}: {model.graph.edges[edge[0]][edge[1]]}"
    )
```

### Supported algorithms
Currently causy supports the following algorithms:
- PC (Peter-Clark)
  - PC - the original PC algorithm without any modifications ```causy.algorithms.PC```
  - ParallelPC - a parallelized version of the PC algorithm ```causy.algorithms.ParallelPC```

### Supported pipeline steps
Detailed information about the pipeline steps can be found in the [API Documentation](https://causy-dev.github.io/causy/causy.html).

## Dev usage

### Setup
We recommend using poetry to manage the dependencies. To install poetry follow the instructions on https://python-poetry.org/docs/#installation.

Install dependencies
```bash
poetry install
```

Execute tests
```bash
poetry run python -m unittest discover -s tests
```
