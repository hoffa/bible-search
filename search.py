import json
import sys

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


def main():
    s = model.encode(sys.argv[1])
    for l in sys.stdin:
        obj = json.loads(l)
        sim = float(util.cos_sim(s, obj["e"]))
        print(f'{sim:.8f}\t{obj["n"]}\t{obj["t"]}')


if __name__ == "__main__":
    main()
