# causy

Causal discovery made easy.

## Dev usage

### Setup
We recommend using poetry to manage the dependencies. To install poetry follow the instructions on https://python-poetry.org/docs/#installation.

```bash
poetry install
```

### Usage

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

Execute tests
```bash
poetry run python -m unittest discover -s tests
```
