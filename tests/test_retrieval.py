from fruity_fun.config import Settings
from fruity_fun.models import Chunk, SearchHit
from fruity_fun.retrieval import HybridRetriever


def make_retriever():
    chunks = [
        Chunk("apple", "Apples grow on trees and can be crisp.", "fruit.pdf", 1),
        Chunk("banana", "Bananas grow in bunches on large herbs.", "fruit.pdf", 2),
    ]
    return HybridRetriever(Settings(_env_file=None), chunks)


def test_bm25_finds_keyword():
    hits = make_retriever().sparse_search("banana bunches")
    assert hits[0].id == "banana"
    assert hits[0].channels == ["bm25"]


def test_rrf_rewards_channel_agreement():
    retriever = make_retriever()
    dense = [SearchHit("apple", "a", "s", 1, 0.9, ["dense"])]
    sparse = [
        SearchHit("banana", "b", "s", 2, 1.0, ["bm25"]),
        SearchHit("apple", "a", "s", 1, 0.8, ["bm25"]),
    ]
    fused = retriever.fuse(dense, sparse)
    assert fused[0].id == "apple"
    assert fused[0].channels == ["bm25", "dense"]
    assert retriever.confidence(fused) > 0.5
