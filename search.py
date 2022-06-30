import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Union

import pandas  # type: ignore
import torch
from sentence_transformers import SentenceTransformer, util  # type: ignore

from common import from_vid

Pathish = Union[Path, str]


def get_model() -> SentenceTransformer:
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


def read_books_df(path: Pathish) -> dict[int, str]:
    df = pandas.read_parquet(path)
    df.set_index("b", inplace=True)
    return {k: v["n"] for k, v in df.to_dict(orient="index").items()}


def read_text_df(path: Pathish) -> dict[int, str]:
    df = pandas.read_parquet(path)
    df.set_index(["vid"], inplace=True)
    return {k: v["t"] for k, v in df.to_dict(orient="index").items()}


def read_embeddings_df(path: Pathish) -> tuple[pandas.DataFrame, torch.Tensor]:
    df = pandas.read_parquet(path)
    tensor = torch.tensor(df["e"])
    return df, tensor


@dataclass
class SearchResult:
    book: str
    chapter: int
    verse: int
    text: str


def get_results_df(
    embeddings_df: pandas.DataFrame,
    query_embedding: torch.Tensor,
    top_k: int,
    embeddings_tensor: torch.Tensor,
) -> Iterator[int]:
    results = util.semantic_search(query_embedding, embeddings_tensor, top_k=top_k)
    for result in results[0]:
        yield embeddings_df.iloc[result["corpus_id"]]["vid"]


def search(books_df: dict, text_df: dict, results_df) -> Iterator[SearchResult]:
    for vid in results_df:
        b, c, v = from_vid(vid)
        book = books_df[b]
        text = text_df[vid]
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
