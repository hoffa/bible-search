import urllib
from pathlib import Path

import streamlit as st

from search import search, get_model, read_books_df, read_text_df, read_embeddings_df

st.set_page_config(page_title="Bible Search", page_icon=":books:")
st.title("Bible Search")


query = st.text_input("Query", placeholder="What is love?")


st.radio("Version", ["World English Bible (WEB)"])


@st.cache
def _get_dfs():
    books_df = read_books_df(
        "https://github.com/hoffa/bible/releases/download/v1/web_books.parquet"
    )
    text_df = read_text_df(
        "https://github.com/hoffa/bible/releases/download/v1/web_text.parquet"
    )
    embeddings_df = read_embeddings_df(
        "https://github.com/hoffa/bible/releases/download/v1/web_embeddings.parquet"
    )
    return books_df, text_df, embeddings_df


@st.cache(allow_output_mutation=True)
def _get_model():
    return get_model()


def get_verse_url(result):
    return "https://www.biblegateway.com/passage/?" + urllib.parse.urlencode(
        {
            "search": f"{result.book} {result.chapter}:{result.verse}",
        },
    )


if query:
    model = _get_model()
    books_df, text_df, embeddings_df = _get_dfs()
    results = search(books_df, text_df, embeddings_df, model.encode(query))
    out = ""
    for result in results:
        out += f"1. [{result.book} {result.chapter}:{result.verse}]({get_verse_url(result)}) - {result.text}\n"
    st.subheader("Results")
    st.write(out)
