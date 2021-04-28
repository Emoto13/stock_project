.PHONY: activate activate_install deactivate run lint clean

create_venv:
	python3 -m venv venv;

activate: create_venv
	(. venv/bin/activate;)

activate_install: activate
	(pip3 install -r requirements.txt; )

deactivate: activate
	(deactivate)

run_main: activate
	python3 src/main.py;
	(make clean);

run_playground: activate
	python3 src/playground.py;
	(make clean)

run_file: activate
	python3 $(path);
	(make clean)

lint:
	flake8 src;

clean:
	find . -type f -name "*.py[co]" -delete;
	find . -type d -name "__pycache__" -delete;
