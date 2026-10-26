"""Static evidence checks. Absence of evidence never means reproduction succeeded."""

from pathlib import Path

from .retrieval import RepositoryIndex


def audit_repository(index: RepositoryIndex) -> dict:
    files = index.files
    checks = [
        ("environment", "requirements python dependencies install", lambda p:
         Path(p).name in {"requirements.txt", "pyproject.toml", "environment.yml",
                          "environment.yaml", "Dockerfile"}),
        ("data", "dataset data download 数据", lambda p:
         p.startswith(("data/", "datasets/")) and Path(p).suffix in {".json", ".txt"}),
        ("entrypoint", "train evaluate inference run python", lambda p:
         Path(p).name in {"train.py", "evaluate.py", "eval.py", "main.py", "run.py"}),
        ("weights", "checkpoint pretrained model weights 权重", lambda p: False),
    ]
    findings = []
    for name, query, detector in checks:
        matches = [p for p in files if detector(p)]
        evidence = index.search(query, k=3)
        findings.append({"check": name, "status": "file_found" if matches else "needs_review",
                         "files": matches, "evidence": evidence,
                         "note": "File presence is not execution or dependency validation."})
    return {"repository": str(index.root), "indexed_files": len(files),
            "chunks": len(index.chunks), "coverage_warnings": index.warnings,
            "findings": findings, "execution_status": "not_run",
            "paper_metric_status": "not_verified"}


def markdown_report(audit: dict, *, question: str = "", answer: str = "") -> str:
    lines = ["# PaperTrail evidence report", "", f"Repository: `{audit['repository']}`", "",
             "**Execution: not run. Paper metrics: not verified.**", "",
             f"Indexed {audit['indexed_files']} files / {audit['chunks']} chunks.", ""]
    for warning in audit["coverage_warnings"]:
        lines += [f"> Coverage warning: {warning}", ""]
    for finding in audit["findings"]:
        lines += [f"## {finding['check']}: {finding['status']}", "", finding["note"], ""]
        for evidence in finding["evidence"]:
            # Quote repository text as inert evidence, not report instructions.
            lines += [f"Source: `{evidence['citation']}` · SHA256 `{evidence['digest'][:12]}`", ""]
            lines += ["> " + line for line in evidence["text"].splitlines()[:12]]
            lines += [""]
    if answer:
        lines += ["## Model-generated analysis (verify against cited evidence)", "", question,
                  "", answer, ""]
    return "\n".join(lines)
