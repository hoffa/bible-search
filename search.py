import argparse
from pathlib import Path
from dataclasses import dataclass

import pandas
from sentence_transformers import SentenceTransformer, util


def get_model():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


def read_dfs(dir):
    books_df = pandas.read_parquet(dir / "books.parquet")
    books_df.set_index("b", inplace=True)

    text_df = pandas.read_parquet(dir / "text.parquet")
    text_df.set_index(["b", "c", "v"], inplace=True)

    embeddings_df = pandas.read_parquet(dir / "embeddings.parquet")

    return books_df, text_df, embeddings_df


@dataclass
class SearchResult:
    book: str
    chapter: int
    verse: int
    text: str


def search(books_df, text_df, embeddings_df, query_embedding, results=100):
    df = embeddings_df.sort_values(
        by="e",
        ascending=False,
        key=lambda col: col.map(lambda e: util.cos_sim(query_embedding, e)),
    ).head(results)[["b", "c", "v"]]
    results = df.to_dict(orient="records")
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

    books_df, text_df, embeddings_df = read_dfs(args.books.parent)

    for result in search(books_df, text_df, embeddings_df, s):
        print(result.book, result.chapter, result.verse, result.text)


if __name__ == "__main__":
    main()
