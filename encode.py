import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TypedDict

import pandas  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
import torch

from common import to_vid

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


@dataclass
class TextRow:
    book: int
    chapter: int
    verse: int
    text: str


@dataclass
class BooksRow:
    book: int
    name: str


class TextDfRow(TypedDict):
    vid: int
    t: str


class BooksDfRow(TypedDict):
    b: int
    n: str


class EmbeddingsDfRow(TypedDict):
    vid: int
    e: list[torch.Tensor]


def _parse_text(path: Path) -> Iterator[TextRow]:
    with path.open() as f:
        for row in csv.DictReader(f):
            yield TextRow(
                book=int(row["b"]),
                chapter=int(row["c"]),
                verse=int(row["v"]),
                text=row["t"],
            )


def _parse_books(path: Path) -> Iterator[BooksRow]:
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            yield BooksRow(
                book=int(row[0]),
                name=row[1],
            )


def read_text(path: Path) -> Iterator[TextDfRow]:
    for row in _parse_text(path):
        vid = to_vid(row.book, row.chapter, row.verse)
        yield {
            "vid": vid,
            "t": row.text,
        }


def read_embeddings(path: Path) -> Iterator[EmbeddingsDfRow]:
    for row in _parse_text(path):
        vid = to_vid(row.book, row.chapter, row.verse)
        yield {
            "vid": vid,
            "e": model.encode(row.text),
        }


def read_books(path: Path) -> Iterator[BooksDfRow]:
    for row in _parse_books(path):
        yield {
            "b": row.book,
            "n": row.name,
        }


def write_parquet(path: Path, data: Any) -> None:
    pandas.DataFrame(data=data).to_parquet(path, compression="gzip")


def main() -> None:
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
