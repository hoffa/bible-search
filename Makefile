VENV = .venv
BIN = $(VENV)/bin
PYTHON = $(BIN)/python

init:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(BIN)/streamlit run app.py

clean:
	rm -rf $(VENV)

data:
	./generate_data.sh