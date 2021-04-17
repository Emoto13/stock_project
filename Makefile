.PHONY: activate activate_install deactivate run lint clean

activate:
	(. venv/bin/activate;)

activate_install: activate
	(pip3 install -r requirements.txt; )

deactivate: activate
	(deactivate)

run: activate
	python3 src/main.py

lint:
	flake8 src

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
