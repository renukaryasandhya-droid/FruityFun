from fruity_fun.corpus import _windows


def test_windows_overlap_and_preserve_content():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = _windows(text, size=600, overlap=120)
    assert len(chunks) > 1
    assert "word0" in chunks[0]
    assert "word399" in chunks[-1]


def test_windows_empty():
    assert _windows("") == []
