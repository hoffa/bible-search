from dataclasses import dataclass

import pandas
import torch
from sentence_transformers import util

from common import from_vid


def read_books_df(path):
    df = pandas.read_parquet(path)
    df.set_index("b", inplace=True)
    return {k: v["n"] for k, v in df.to_dict(orient="index").items()}


def read_text_df(path):
    df = pandas.read_parquet(path)
    df.set_index(["vid"], inplace=True)
    return {k: v["t"] for k, v in df.to_dict(orient="index").items()}


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


def get_results_df(
    embeddings_df,
    query_embedding,
    top_k,
    embeddings_tensor,
):
    results = util.semantic_search(query_embedding, embeddings_tensor, top_k=top_k)
    for result in results[0]:
        yield embeddings_df.iloc[result["corpus_id"]]["vid"]


def search(
    books_df,
    text_df,
    results_df,
):
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
