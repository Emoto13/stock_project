.ONESHELL:
.PHONY: create_venv create_install run_main run_playground run_file lint clean

BIN=venv/bin/

create_venv:
	python3 -m venv venv;

create_install: create_venv
	$(BIN)pip3 install -r requirements.txt

run_main: create_install
	$(BIN)python3 src/main.py;
	make clean

run_playground: create_install
	$(BIN)python3 src/playground.py
	make clean

run_file: activate
	$(BIN)python3 $(path);
	make clean

lint:
	flake8 src;

clean:
	find . -type f -name "*.py[co]" -delete;
	find . -type d -name "__pycache__" -delete;
