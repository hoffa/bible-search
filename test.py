import unittest
from common import get_model

from search import (
    get_results_df,
    read_books_df,
    read_embeddings_df,
    read_text_df,
    search,
)


class Test(unittest.TestCase):
    def test_search(self) -> None:
        books = read_books_df("dist/web_books.parquet")
        text = read_text_df("dist/web_text.parquet")
        embeddings, embeddings_tensor = read_embeddings_df(
            "dist/web_embeddings.parquet"
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
