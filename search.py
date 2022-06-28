import argparse
from pathlib import Path
from dataclasses import dataclass

import pandas
import torch
from sentence_transformers import SentenceTransformer, util

from common import from_vid


def get_model():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


def read_books_df(path):
    df = pandas.read_parquet(path)
    df.set_index("b", inplace=True)
    return df.to_dict(orient="index")


def read_text_df(path):
    df = pandas.read_parquet(path)
    df.set_index(["vid"], inplace=True)
    return df.to_dict(orient="index")


def read_embeddings_df(path):
    df = pandas.read_parquet(path)
    tensor = torch.tensor(df["e"])
    return df, tensor


@dataclass
class SearchResult:
    book: str
    chapter: int
    verse: int
    text: str


def get_results_df(embeddings_df, query_embedding, results, embeddings_tensor):
    results = util.semantic_search(query_embedding, embeddings_tensor, top_k=results)
    for result in results[0]:
        yield embeddings_df.iloc[result["corpus_id"]]["vid"]


def search(books_df, text_df, results_df):
    for vid in results_df:
        b, c, v = from_vid(vid)
        book = books_df[b]["n"]
        text = text_df[vid]["t"]
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
