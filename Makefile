VENV = .venv
BIN = $(VENV)/bin
PYTHON = $(BIN)/python

init:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install --upgrade -r requirements.txt
	$(PYTHON) -m pip install --upgrade black pyflakes mypy

format:
	$(PYTHON) -m black .

test:
	$(PYTHON) -m unittest test.py
	$(PYTHON) -m black --check .
	$(PYTHON) -m pyflakes *.py
	$(PYTHON) -m mypy --strict *.py

run:
	$(BIN)/streamlit run app.py

clean:
	rm -rf $(VENV)

data:
	./data.sh
