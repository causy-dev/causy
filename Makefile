test:
	poetry run python -m unittest

coverage:
	poetry run coverage run -m unittest
	poetry run coverage report -m
