import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pandas
import torch

from common import get_model, to_vid

model = get_model()


@dataclass
class TextRow:
    text: str
    vid: int
    book: int
    chapter: int
    verse: int


class TextDfRow(TypedDict):
    vid: int
    t: str


class BooksDfRow(TypedDict):
    b: int
    n: str


class EmbeddingsDfRow(TypedDict):
    vid: int
    e: list[torch.Tensor]


def _parse_text(path):
    with path.open() as f:
        for row in csv.DictReader(f):
            b = int(row["b"])
            c = int(row["c"])
            v = int(row["v"])
            vid = to_vid(b, c, v)
            yield TextRow(
                text=row["t"],
                vid=vid,
                book=b,
                chapter=c,
                verse=v,
            )


def read_text(path):
    for row in _parse_text(path):
        yield TextDfRow(
            vid=row.vid,
            t=row.text,
        )


def read_embeddings(path):
    for row in _parse_text(path):
        yield EmbeddingsDfRow(
            vid=row.vid,
            e=model.encode(row.text),
        )


def read_books(path):
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            b = int(row[0])
            n = row[1]
            yield BooksDfRow(
                b=b,
                n=n,
            )


def write_parquet(path, data):
    pandas.DataFrame(data=data).to_parquet(path, compression="gzip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-in", required=True, type=Path)
    parser.add_argument("--books-in", required=True, type=Path)
    parser.add_argument("--text-out", required=True, type=Path)
    parser.add_argument("--books-out", required=True, type=Path)
    parser.add_argument("--embeddings-out", required=True, type=Path)
    args = parser.parse_args()

    print(f"Writing {args.books_out}")
    write_parquet(args.books_out, read_books(args.books_in))

    print(f"Writing {args.text_out}")
    write_parquet(args.text_out, read_text(args.text_in))

    print(f"Writing {args.embeddings_out}")
    write_parquet(args.embeddings_out, read_embeddings(args.text_in))


if __name__ == "__main__":
    main()
