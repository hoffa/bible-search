from sentence_transformers import SentenceTransformer


def to_vid(b, c=0, v=0):
    return (b * 1000000) + (c * 1000) + v


def from_vid(vid):
    b = vid // 1000000
    c = (vid % 1000000) // 1000
    v = vid % 1000
    return b, c, v


def get_model():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
