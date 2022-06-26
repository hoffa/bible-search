#!/bin/sh
set -eux

python3 -m venv .venv
PYTHON="${PWD}/.venv/bin/python"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install -r requirements.txt

mkdir -p dist
cd dist

curl -O https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_web.csv
curl -O https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_bbe.csv
curl -O https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_kjv.csv
curl -O https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_ylt.csv
curl -O https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/key_english.csv

# $1 - translation key
generate_scrollmapper() {
    "${PYTHON}" ../encode_scrollmapper.py --text-in "t_$1.csv" --books-in key_english.csv --text-out "$1_text.parquet" --books-out "$1_books.parquet" --embeddings-out "$1_embeddings.parquet"
}

generate_scrollmapper web
generate_scrollmapper bbe
generate_scrollmapper kjv
generate_scrollmapper ylt