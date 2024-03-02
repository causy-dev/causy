"""
Representation of a causy workspace.

## Exemplary folder structure
- causy.yaml (contains workspace wide settings for stuff like name, author, pipeline definitions)
- any_pipeline.yaml (seperate pipeline definition referenced by causy.yaml)
- custom_pipeline_step.py (a project specific pipeline step)
- some_data_loader.py (a data loader to ship data into causy)
- tune_pipeline_1709406402/ (a single experiment)
    - causy_experiment.json (contains the full definition of everything used)
    - causy_model.json (the actual generated model)

## Examples
- the user generates a new workspace because they want to work on a new causal discovery project (causy workspace create)
- they add a custom data loader, pipeline steps and extend an existing pipeline (causy workspace data-loaders create)
- they run multiple experiments (causy workspace experiments create, causy workspace experiments execute $EXPERIMENT_NAME)
"""
