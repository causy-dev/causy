# causy

Causal discovery made easy.

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

### Usage via CLI

Run causy with one of the default algorithms
```bash
poetry run causy execute --help
poetry run causy execute tests/fixtures/toy_data_larger.json --algorithm PC
```

Customize your causy pipeline by ejecting and modifying the pipeline file.
```bash
poetry run causy eject PC pc.json
# edit pc.json
poetry run causy execute tests/fixtures/toy_data_larger.json --pipeline pc.json
```


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
