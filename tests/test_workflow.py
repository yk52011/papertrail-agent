import json
from pathlib import Path

import pytest

from papertrail.jobs import JobStore
from papertrail.workflow import resume_job, start_job


def fixture_repository(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Fix train_path to data.json for FileNotFoundError.\n")
    (source / "data.json").write_text('[{"id":"a","label":1},{"id":"b","label":0}]')
    (source / "config.json").write_text('{"train_path":"missing.json"}')
    (source / "task.json").write_text(json.dumps({"title": "test recovery",
                                                 "allowed_updates": {"train_path": ["data.json"]}}))
    (source / "experiment.json").write_text(json.dumps({"command": ["python", "train.py"],
        "artifact": "metrics.json", "metric": "accuracy", "minimum": .8,
        "ground_truth": "data.json", "timeout_seconds": 5}))
    (source / "train.py").write_text('''import json
from pathlib import Path
config = json.loads(Path('config.json').read_text())
rows = json.loads(Path(config['train_path']).read_text())
Path('metrics.json').write_text(json.dumps({'accuracy':1.0,
  'predictions':[{'id':r['id'],'prediction':r['label']} for r in rows]}))
''')
    return source, JobStore(tmp_path / "jobs")


def test_closed_loop_repairs_only_snapshot_and_recomputes_metric(tmp_path):
    source, store = fixture_repository(tmp_path)
    state = start_job(source, store=store, backend="local")
    assert state["status"] == "succeeded"
    assert len(state["attempts"]) == 2
    assert len(state["patches"]) == 1
    assert state["attempts"][-1]["verification"]["independent"]["accuracy"] == 1
    assert json.loads((source / "config.json").read_text())["train_path"] == "missing.json"
    assert store.load(state["job_id"])["status"] == "succeeded"
    assert [e["node"] for e in state["events"]].count("execute") == 2


def test_resume_from_disk_at_dynamic_interrupt(tmp_path):
    source, store = fixture_repository(tmp_path)
    first = start_job(source, store=store, backend="local", step_mode=True)
    assert first["status"] == "paused"
    assert len(first["attempts"]) == 1 and not first["patches"]
    # New store and newly built graph; no in-memory state reused.
    second = resume_job(first["job_id"], store=JobStore(store.root))
    assert second["status"] == "succeeded"
    assert len(second["attempts"]) == 2 and len(second["patches"]) == 1


def test_cancel_checkpoint_does_not_apply_patch(tmp_path):
    source, store = fixture_repository(tmp_path)
    first = start_job(source, store=store, backend="local", step_mode=True)
    second = resume_job(first["job_id"], proceed=False, store=store)
    assert second["status"] == "cancelled"
    assert not second["patches"]


@pytest.mark.parametrize("update", [{"minimum": 0}, {"train_path": "../../secret"}])
def test_llm_proposal_cannot_expand_permissions(tmp_path, update):
    source, store = fixture_repository(tmp_path)

    def planner(state):
        return {"failure_type": "missing_data", "updates": update, "reason": "test",
                "citations": [state["evidence"][0]["citation"]]}

    state = start_job(source, store=store, backend="local", policy="llm", planner=planner)
    assert state["status"] == "failed" and not state["patches"]
    assert "Rejected" in state["stop_reason"]


def test_llm_recovery_requires_real_retrieved_citation(tmp_path):
    source, store = fixture_repository(tmp_path)
    state = start_job(source, store=store, backend="local", policy="llm", planner=lambda s:
        {"failure_type": "missing_data", "updates": {"train_path": "data.json"},
         "reason": "unsupported", "citations": ["nonexistent.md:1-2"]})
    assert state["status"] == "failed" and not state["patches"]


def test_actual_training_workload(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples/sentiment-lab"
    state = start_job(source, store=JobStore(tmp_path / "jobs"), backend="local", max_seconds=120)
    assert state["status"] == "succeeded"
    assert len(state["attempts"]) == 3
    verified = state["attempts"][-1]["verification"]
    assert verified["independent"]["count"] == 20
    assert verified["independent"]["macro_f1"] >= .75
