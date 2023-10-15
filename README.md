# causy

Causal discovery made easy.

## Dev usage

### Setup
We recommend using poetry to manage the dependencies. To install poetry follow the instructions on https://python-poetry.org/docs/#installation.

```bash
poetry install
```

Run causy
```bash
poetry run causy excute --help
poetry run causy excute pipelines/pc.json tests/fixtures/rki.json
```

Execute tests
```bash
poetry run python -m unittest discover -s tests
```
