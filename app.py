import urllib
from pathlib import Path

import streamlit as st

from search import search, get_model, read_books_df, read_text_df, read_embeddings_df

st.set_page_config(page_title="Bible Search", page_icon=":books:")
st.title("Bible Search")


query = st.text_input("Search", placeholder="What is love?")

WEB_NAME = "World English Bible (WEB)"
KJV_NAME = "King James Version (KJV)"
YLT_NAME = "Young's Literal Translation (YLT)"

VERSIONS = {
    WEB_NAME: "web",
    KJV_NAME: "kjv",
    YLT_NAME: "ylt",
}

version = VERSIONS[
    st.radio(
        "Version",
        VERSIONS.keys(),
    )
]


@st.cache
def _get_dfs(version):
    books_df = read_books_df(
        f"https://github.com/hoffa/bible-search/releases/download/v1/{version}_books.parquet"
    )
    text_df = read_text_df(
        f"https://github.com/hoffa/bible-search/releases/download/v1/{version}_text.parquet"
    )
    embeddings_df = read_embeddings_df(
        f"https://github.com/hoffa/bible-search/releases/download/v1/{version}_embeddings.parquet"
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
    with st.spinner("Loading..."):
        model = _get_model()
        books_df, text_df, embeddings_df = _get_dfs(version)
        results = search(books_df, text_df, embeddings_df, model.encode(query))
    out = ""
    for result in results:
        out += f"1. [{result.book} {result.chapter}:{result.verse}]({get_verse_url(result)}) - {result.text}\n"
    st.subheader("Results")
    st.write(out)
