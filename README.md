# bible-search

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hoffa-bible-search-app-1uj1m3.streamlitapp.com)

Semantically search the Bible.

## Data format

Bible data is stored in 3 Parquet files:

1. The Bible text with columns `b`, `c`, `v` and `t`.
2. Verse embeddings with columns `b`, `c`, `v` and `e`.
3. Book number-to-name mapping with columns `b` and `n`.

Where:

- `b` is the book number.
- `n` is the book name.
- `c` is the chapter number.
- `v` is the verse number.
- `t` is the verse text.
- `e` is the verse embedding.

## Generating the data

```bash
./generate_data.sh
```
