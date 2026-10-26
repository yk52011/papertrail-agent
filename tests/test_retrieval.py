import pytest

from papertrail.retrieval import RepositoryIndex


def test_retrieval_returns_grounded_lines_and_skips_secrets(tmp_path):
    (tmp_path / "README.md").write_text("intro\nCUDA requires GPU memory\n")
    (tmp_path / ".env").write_text("secret_key=not-real")
    (tmp_path / "credentials.json").write_text('{"key":"not-real"}')
    (tmp_path / "link.md").symlink_to(tmp_path / "README.md")
    index = RepositoryIndex(tmp_path)
    hits = index.search("CUDA GPU")
    assert hits[0]["citation"] == "README.md:1-2"
    assert index.files == ["README.md"]
    assert index.search("unmatchable") == []
    with pytest.raises(ValueError):
        index.search("x", mode="hybrid")


def test_index_budget_is_visible(tmp_path):
    for i in range(3):
        (tmp_path / f"{i}.md").write_text("test")
    index = RepositoryIndex(tmp_path, max_files=1)
    assert len(index.files) == 1
    assert "incomplete" in index.warnings[0]


class Embeddings:
    def embed_documents(self, texts):
        return [[1, 0] if "GPU" in t else [0, 1] for t in texts]

    def embed_query(self, text):
        return [1, 0]


def test_dense_and_hybrid_paths_without_paid_api(tmp_path):
    (tmp_path / "hardware.md").write_text("GPU memory")
    (tmp_path / "data.md").write_text("dataset labels")
    index = RepositoryIndex(tmp_path, embeddings=Embeddings())
    assert index.search("accelerator", mode="dense")[0]["source"] == "hardware.md"
    assert index.search("GPU", mode="hybrid")[0]["source"] == "hardware.md"
