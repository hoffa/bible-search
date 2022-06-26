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
    return df


def read_text_df(path):
    df = pandas.read_parquet(path)
    df.set_index(["b", "c", "v"], inplace=True)
    return df


def read_embeddings_df(path):
    return pandas.read_parquet(path)


@dataclass
class SearchResult:
    book: str
    chapter: int
    verse: int
    text: str


def _search(embeddings_df, query_embedding, results):
    return embeddings_df.sort_values(
        by="e",
        ascending=False,
        key=lambda col: col.map(lambda e: util.cos_sim(query_embedding, e)),
    ).head(results)[["b", "c", "v"]]


def search(books_df, text_df, embeddings_df, query_embedding, results=100):
    results = _search(embeddings_df, query_embedding, results).to_dict(orient="records")
    for result in results:
        b = result["b"]
        c = result["c"]
        v = result["v"]
        book = books_df.loc[b]["n"]
        text = text_df.loc[(b, c, v)]["t"]
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
