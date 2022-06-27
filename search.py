import argparse
from pathlib import Path
from dataclasses import dataclass

import pandas
from sentence_transformers import SentenceTransformer, util


def get_model():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


def read_books_df(path):
    df = pandas.read_parquet(path)
    df.set_index("b", inplace=True)
    return df.to_dict(orient="index")


def read_text_df(path):
    df = pandas.read_parquet(path)
    df.set_index(["b", "c", "v"], inplace=True)
    return df.to_dict(orient="index")


def read_embeddings_df(path):
    return pandas.read_parquet(path)


@dataclass
class SearchResult:
    book: str
    chapter: int
    verse: int
    text: str


def get_results_df(embeddings_df, query_embedding, results=100):
    embeddings_df["s"] = embeddings_df["e"].apply(
        lambda e: float(util.cos_sim(e, query_embedding))
    )
    return embeddings_df.nlargest(results, "s")[["b", "c", "v", "s"]]


def search(books_df, text_df, results_df):
    for result in results_df.to_dict(orient="records"):
        b = result["b"]
        c = result["c"]
        v = result["v"]
        book = books_df[b]["n"]
        text = text_df[(b, c, v)]["t"]
        yield SearchResult(
            book=book,
            chapter=c,
            verse=v,
            text=text,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--books", required=True, type=Path)
    parser.add_argument("--text", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    model = get_model()
    s = model.encode(args.query)

    books_df = read_books_df(args.books)
    text_df = read_text_df(args.text)
    embeddings_df = read_embeddings_df(args.embeddings)

    for result in search(books_df, text_df, embeddings_df, s):
        print(result.book, result.chapter, result.verse, result.text)


if __name__ == "__main__":
    main()
