# :books: bible-search

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hoffa-bible-search-app-1uj1m3.streamlitapp.com)

Semantically search the Bible. Try it out [here](https://hoffa-bible-search-app-1uj1m3.streamlitapp.com)

Originally based on a [similar idea](https://github.com/chrislee973/bible-semantic-search) by Chris Lee.

Public domain Bible translations courtesy of [Scrollmapper](https://github.com/scrollmapper/bible_databases).

## Development

Set up dependencies:

```bash
make init
```

Run the app:

```bash
make run
```

## Generating the data

```bash
make data
```

The latest data is uploaded to [releases](https://github.com/hoffa/bible-search/releases).

## Data format

Bible data is stored in 3 Parquet files:

1. The Bible text with columns `vid` and `t`.
2. Verse embeddings with columns `vid` and `e`.
3. Book number-to-name mapping with columns `b` and `n`.

Where:

- `vid` is the verse ID. It's an integer in the format `bbbcccvvv` where `bbb` is the book number, `ccc` is the chapter number, and `vvv` is the verse number.
- `b` is the book number.
- `n` is the book name.
- `t` is the verse text.
- `e` is the verse embedding.
