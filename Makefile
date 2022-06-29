VENV = .venv
BIN = $(VENV)/bin
PYTHON = $(BIN)/python
PROD = requirements/prod.txt
DEV = requirements/dev.txt

init:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt

dev:
	$(PYTHON) -m pip install -r requirements/dev.txt

run:
	$(BIN)/streamlit run app.py

clean:
	rm -rf $(VENV)

data:
	./generate_data.sh

requirements: clean init
	$(PYTHON) -m pip freeze > requirements.txt
