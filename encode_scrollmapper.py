import argparse
import csv
from pathlib import Path

import pandas
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


def read_text(path):
    with path.open() as f:
        for row in csv.DictReader(f):
            b = int(row["b"])
            c = int(row["c"])
            v = int(row["v"])
            t = row["t"]
            yield {
                "b": b,
                "c": c,
                "v": v,
                "t": t,
            }


def read_embeddings(path):
    with path.open() as f:
        for row in csv.DictReader(f):
            b = int(row["b"])
            c = int(row["c"])
            v = int(row["v"])
            e = model.encode(row["t"]).tolist()
            yield {
                "b": b,
                "c": c,
                "v": v,
                "e": e,
            }


def read_books(path):
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            yield {
                "b": int(row[0]),
                "n": row[1],
            }


def write_parquet(path, data):
    pandas.DataFrame(data=data).to_parquet(path, compression="gzip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, type=Path)
    parser.add_argument("--books", required=True, type=Path)
    parser.add_argument("--dir", required=True, type=Path)
    args = parser.parse_args()

    args.dir.mkdir(parents=True, exist_ok=True)

    print("Writing books.parquet")
    write_parquet(args.dir / "books.parquet", read_books(args.books))

    print("Writing text.parquet")
    write_parquet(args.dir / "text.parquet", read_text(args.text))

    print("Writing embeddings.parquet")
    write_parquet(args.dir / "embeddings.parquet", read_embeddings(args.text))


if __name__ == "__main__":
    main()
