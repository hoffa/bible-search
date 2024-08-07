VENV = .venv
BIN = $(VENV)/bin
PYTHON = $(BIN)/python

init:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install --upgrade -r requirements.txt
	mkdir -p dist && \
	cd dist && \
	curl -O https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_web.csv && \
	curl -O https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_bbe.csv && \
	curl -O https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_kjv.csv && \
	curl -O https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_ylt.csv && \
	curl -O https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/key_english.csv

test:
	$(PYTHON) -m unittest test.py

run:
	$(BIN)/streamlit run app.py

clean:
	rm -rf $(VENV)

data:
	./data.sh
