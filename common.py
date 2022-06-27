def to_vid(b, c, v):
    return (b * 1000000) + (c * 1000) + v


def from_vid(vid):
    b = vid // 1000000
    c = (vid % 1000000) // 1000
    v = vid % 1000
    return b, c, v
