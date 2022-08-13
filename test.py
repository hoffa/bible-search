from pathlib import Path
import unittest
from common import from_vid, get_model, to_vid
from encode import read_books, read_embeddings, read_text

from search import (
    get_results_df,
    read_books_df,
    read_embeddings_df,
    read_text_df,
    search,
)


def path_or_url(path: str, url: str) -> str:
    return path if Path(path).exists() else url


class Test(unittest.TestCase):
    def test_search(self) -> None:
        books = read_books_df(
            path_or_url(
                "dist/web_books.parquet",
                "https://github.com/hoffa/bible-search/releases/download/v1/web_books.parquet",
            )
        )
        text = read_text_df("dist/web_text.parquet")
        embeddings, embeddings_tensor = read_embeddings_df(
            path_or_url(
                "dist/web_embeddings.parquet",
                "https://github.com/hoffa/bible-search/releases/download/v1/web_embeddings.parquet",
            )
        )
        model = get_model()
        vids = get_results_df(
            embeddings, model.encode("what is love"), 50, embeddings_tensor
        )
        results = list(search(books, text, vids))
        self.assertEqual(
            results[4].text,
            "Love is patient and is kind; love doesn't envy. Love doesn't brag, is not proud,",
        )

    def test_encode(self) -> None:
        self.assertEqual(
            list(
                read_text(
                    Path(
                        path_or_url(
                            "dist/t_web.csv",
                            "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_web.csv",
                        )
                    )
                )
            )[42],
            {
                "vid": 1002012,
                "t": "and the gold of that land is good. There is aromatic resin and the onyx stone.",
            },
        )
        self.assertEqual(
            list(
                read_books(
                    Path(
                        path_or_url(
                            "dist/key_english.csv",
                            "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/key_english.csv",
                        )
                    )
                )
            )[10],
            {"b": 11, "n": "1 Kings"},
        )
        embedding = next(read_embeddings(Path("dist/t_web.csv")))
        self.assertEqual(embedding["vid"], 1001001)

    def test_vid(self) -> None:
        self.assertEqual(to_vid(999, 999, 999), 999_999_999)
        self.assertEqual(to_vid(1, 23, 456), 1_023_456)
        self.assertEqual(to_vid(1, 23), 1_023_000)
        self.assertEqual(to_vid(12), 12_000_000)
        self.assertEqual(from_vid(999_999_999), (999, 999, 999))
        self.assertEqual(from_vid(1_023_456), (1, 23, 456))
        self.assertEqual(from_vid(1_023_000), (1, 23, 0))
        self.assertEqual(from_vid(12_000_000), (12, 0, 0))
