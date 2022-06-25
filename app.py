import urllib
from pathlib import Path

import streamlit as st

from search import search, get_model, read_dfs

st.title("Bible Search")

query = st.text_input("Search for anything in the Bible")

books_df, text_df, embeddings_df = read_dfs(Path("web"))
model = get_model()


def get_verse_url(result):
    return "https://www.biblegateway.com/passage/?" + urllib.parse.urlencode(
        {
            "search": f"{result.book} {result.chapter}:{result.verse}",
        },
    )


if query:
    out = ""
    for result in search(books_df, text_df, embeddings_df, model.encode(query)):
        out += f"1. [{result.book} {result.chapter}:{result.verse}]({get_verse_url(result)}) - {result.text}\n"
    st.write(out)
