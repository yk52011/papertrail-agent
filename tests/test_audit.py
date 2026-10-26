from papertrail.audit import audit_repository, markdown_report
from papertrail.retrieval import RepositoryIndex
from papertrail.store import save_run


def test_audit_does_not_claim_reproduction(tmp_path):
    (tmp_path / "requirements.txt").write_text("torch")
    result = audit_repository(RepositoryIndex(tmp_path))
    assert result["findings"][0]["status"] == "file_found"
    assert result["execution_status"] == "not_run"
    assert result["paper_metric_status"] == "not_verified"
    assert "not verified" in markdown_report(result)
    assert len(save_run(result, tmp_path / "history")) == 32
