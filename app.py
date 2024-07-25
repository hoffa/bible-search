import urllib

import streamlit as st
from common import get_model

from search import (
    search,
    read_books_df,
    read_text_df,
    read_embeddings_df,
    get_results_df,
)

st.set_page_config(page_title="Bible Search", page_icon=":books:")
st.title("Bible Search")


query = st.text_input("Search", placeholder="What is love?")

WEB_NAME = "World English Bible"
KJV_NAME = "King James Version"
YLT_NAME = "Young's Literal Translation"
BBE_NAME = "Bible in Basic English"

VERSIONS = {
    WEB_NAME: "web",
    KJV_NAME: "kjv",
    YLT_NAME: "ylt",
    BBE_NAME: "bbe",
}

name = st.radio("Translation", VERSIONS.keys())
version = VERSIONS[name or WEB_NAME]


@st.cache(show_spinner=False)
def get_bible(version):
    books_df = read_books_df(
        f"https://github.com/hoffa/bible-search/releases/download/v1/{version}_books.parquet"
    )
    text_df = read_text_df(
        f"https://github.com/hoffa/bible-search/releases/download/v1/{version}_text.parquet"
    )
    embeddings_df, embeddings_tensor = read_embeddings_df(
        f"https://github.com/hoffa/bible-search/releases/download/v1/{version}_embeddings.parquet"
    )
    return books_df, text_df, embeddings_df, embeddings_tensor


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_transformer():
    return get_model()


def get_verse_url(result, version):
    # BBE isn't on BibleGateway
    version = "niv" if version == "bbe" else version
    return "https://www.biblegateway.com/passage/?" + urllib.parse.urlencode(
        {
            "search": f"{result.book} {result.chapter}",
            "version": version,
        },
    )


if query:
    with st.spinner("Loading..."):
        model = get_transformer()
        books_df, text_df, embeddings_df, embeddings_tensor = get_bible(version)
        results_df = get_results_df(
            embeddings_df, model.encode(query), 100, embeddings_tensor
        )
        results = search(books_df, text_df, results_df)
    st.subheader("Results")
    for result in results:
        with st.expander(
            f"{result.book} {result.chapter}:{result.verse}", expanded=True
        ):
            st.write(
                f"{result.text}\n\n[Read {result.book} {result.chapter}]({get_verse_url(result, version)})"
            )

st.caption(
    "This app allows you to search the Bible [semantically](https://en.wikipedia.org/wiki/Semantic_search). The source code is available [here](https://github.com/hoffa/bible-search)."
)
