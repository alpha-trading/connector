all:
	poetry install -D

setup:
	dephell deps convert --from=pyproject.toml --to=setup.py


