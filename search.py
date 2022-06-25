import argparse
import json
import sys
from pathlib import Path

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

    # sort by cosine similarity against query
    verses = (
        embeddings_df.sort_values(
            by="e",
            key=lambda col: col.map(lambda e: util.cos_sim(s, e)),
            ascending=False,
        )
        .head(10)[["b", "c", "v"]]
        .to_dict(orient="records")
    )

    for verse in verses:
        a = books_df.loc[verse["b"]]["n"]
        c = text_df.loc[(verse["b"], verse["c"], verse["v"])]["t"]
        print(f'{a} {verse["c"]}:{verse["v"]} - {c}')

    # apply cosine similarity to each embedding
    # cosine_similarity = util.batch_dot(s, embeddings.values)

    # get top 100 util.cos_sim() rows
    # print(embeddings.sort_values("e", ascending=False).head(100))

    # s = model.encode(sys.argv[1])
    # for l in sys.stdin:
    #    obj = json.loads(l)
    #    sim = float(util.cos_sim(s, obj["e"]))
    #    print(f'{sim:.8f}\t{obj["n"]}\t{obj["t"]}')


if __name__ == "__main__":
    main()
