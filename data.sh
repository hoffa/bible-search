#!/bin/sh
set -eux

PYTHON="${PWD}/.venv/bin/python"

cd dist

# $1 - translation key
generate_scrollmapper() {
	"${PYTHON}" ../encode.py --text-in "t_$1.csv" --books-in key_english.csv --text-out "$1_text.parquet" --books-out "$1_books.parquet" --embeddings-out "$1_embeddings.parquet"
}

generate_scrollmapper web
generate_scrollmapper bbe
generate_scrollmapper kjv
generate_scrollmapper ylt
