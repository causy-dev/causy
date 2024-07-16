> [!WARNING]
> causy is a prototype. Please report any issues and be mindful when using it in production.
# causy

causy is a command line tool that allows you to apply causal inference methods like causal discovery and causal effect estimation. You can adjust causal discovery algorithms with easy to use, extend and maintain pipelines. causy is built based on pytorch which allows you to run the algorithms on CPUs as well as GPUs.

causy workspaces allow you to manage your data sets, algorithm adjustments, and (hyper-)parameters for your experiments.

causy UI allows you to look at your resulting graphs in the browser and gain further insights into every step of the algorithms.

You can find the documentation [here](https://causy.dev/use/).

## Installation
Currently, we support python 3.11 and 3.12. To install causy run
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
Funded by the Prototype Fund from March 2024 until September 2024

![pf_funding_logos](https://github.com/causy-dev/causy/assets/94297994/4d8e4b18-dbe0-4549-bf7e-71f8bd24fdac)
