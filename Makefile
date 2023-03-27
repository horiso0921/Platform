.PHONY: format
format:
	pipenv sync
	pipenv install --skip-lock --dev --ignore-pipfile
	pipenv run autopep8 -ri src
	pipenv run autopep8 -ri tests
	pipenv run isort -rc src
	pipenv run isort -rc tests
	pipenv run yapf src -r -i
	pipenv run yapf tests -r -i