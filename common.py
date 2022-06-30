from sentence_transformers import SentenceTransformer  # type: ignore


def to_vid(b: int, c: int, v: int) -> int:
    return (b * 1000000) + (c * 1000) + v


def from_vid(vid: int) -> tuple[int, int, int]:
    b = vid // 1000000
    c = (vid % 1000000) // 1000
    v = vid % 1000
    return b, c, v


def get_model() -> SentenceTransformer:
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
