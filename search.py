from dataclasses import dataclass
from typing import Iterable, Iterator

import pandas  # type: ignore
import torch
from sentence_transformers import util  # type: ignore

from common import from_vid


def read_books_df(path: str) -> dict[int, str]:
    df = pandas.read_parquet(path)
    df.set_index("b", inplace=True)
    return {k: v["n"] for k, v in df.to_dict(orient="index").items()}


def read_text_df(path: str) -> dict[int, str]:
    df = pandas.read_parquet(path)
    df.set_index(["vid"], inplace=True)
    return {k: v["t"] for k, v in df.to_dict(orient="index").items()}


def read_embeddings_df(path: str) -> tuple[pandas.DataFrame, torch.Tensor]:
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


def search(
    books_df: dict[int, str],
    text_df: dict[int, str],
    results_df: Iterable[int],
) -> Iterator[SearchResult]:
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
