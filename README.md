> [!WARNING]
> causy is currently in a very early and experimental stage of development. We do not recommend using it in production.
# causy

Causal discovery made easy. Causy allows you to use and implement causal discovery algorithms with easy to use, extend and maintain pipelines. It is built based on pytorch which allows you to run the algorithms on CPUs as well as GPUs seamlessly.

## Background

Current causal discovery algorithms are often designed for the primary purpose of research. They are often implemented in a monolithic way, which makes it hard to understand and extend them. Causy aims to solve this problem by providing a framework which allows you to easily implement and use causal discovery algorithms by splitting them up into smaller logic steps which can be stacked together to form a pipeline. This allows you to easily understand, extend, optimize, and experiment with the algorithms.

By shipping causy with sensitively configured default pipelines, we also aim to provide a tool that can be used by non-experts to get started with causal discovery.

Causy workspaces allow you to easily manage your experiments and visualize the causal discovery process. Causy can also be used via code for researchers who want to quickly test adjustments to a causal discovery algorithm by modifying a pipeline.

## Installation
Currently, we only support python 3.11. To install causy run
```bash
pip install causy
```

## Usage
Causy can be used with workspaces via CLI or via code. 

### Workspaces Quickstart

See options for causy workspace
```bash
causy workspace --help
```

Create a new workspace and start the process to configure your pipeline, data loader and experiments interactively. Your  input data should be a json file stored in the same directory. 
```bash
causy workspace init
```

Add an experiment 
```bash
causy workspace experiment add your_experiment_name
```

Update a variable in the experiment
```bash
causy workspace experiment update-variable your_experiment_name your_variable_name your_variable_value 
```

Run multiple experiments
```bash
causy workspace execute 
```

Compare the graphs of the experiments with different variable values via a matrix plot
```bash
causy workspace diff
```

Compare the graphs in the UI, switch between different experiments and visualize the causal discovery process
```bash
causy ui
```

### Usage via Code

Use a default algorithm

```python
from causy.algorithms import PC
from causy.graph_utils import retrieve_edges

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
from causy.common_pipeline_steps.exit_conditions import ExitOnNoActions
from causy.graph_model import graph_model_factory
from causy.common_pipeline_steps.logic import Loop
from causy.common_pipeline_steps.calculation import CalculatePearsonCorrelations
from causy.independence_tests.common import (
  CorrelationCoefficientTest,
  PartialCorrelationTest,
  ExtendedPartialCorrelationTestMatrix,
)
from causy.orientation_rules.pc import (
  ColliderTest,
  NonColliderTest,
  FurtherOrientTripleTest,
  OrientQuadrupleTest,
  FurtherOrientQuadrupleTest,
)
from causy.graph_utils import retrieve_edges

CustomPC = graph_model_factory(
  pipeline_steps=[
    CalculatePearsonCorrelations(),
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
Currently, causy supports the following algorithms:
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
poetry run python -m unittest
```
Funded by Prototype Fund from March 2024 until September 2024

![pf_funding_logos](https://github.com/causy-dev/causy/assets/94297994/4d8e4b18-dbe0-4549-bf7e-71f8bd24fdac)
