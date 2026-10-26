import json
import sys
from pathlib import Path

import pytest

from papertrail.runner import Experiment, run_experiment, verify_artifact


def manifest(root, code, **kwargs):
    spec = {"command": [sys.executable, "-c", code], "artifact": "metrics.json",
            "metric": "accuracy", "minimum": .8, "timeout_seconds": 5, **kwargs}
    path = root / "experiment.json"
    path.write_text(json.dumps(spec))
    return path


def test_exit_zero_without_artifact_is_not_success(tmp_path):
    path = manifest(tmp_path, "print('success')")
    # A stale pre-existing metric must not yield false success.
    (tmp_path / "metrics.json").write_text('{"accuracy":1}')
    result = run_experiment(tmp_path, path, backend="local")
    assert result["exit_code"] == 0
    assert not result["verification"]["verified"]
    assert (tmp_path / "metrics.json").exists()  # Original repository untouched.


def test_local_run_verifies_fresh_artifact(tmp_path):
    path = manifest(tmp_path, "import json; open('metrics.json','w').write(json.dumps({'accuracy':1}))")
    result = run_experiment(tmp_path, path, backend="local")
    assert result["verification"]["verified"]
    assert result["paper_metric_status"] == "not_verified"
    assert not (tmp_path / "metrics.json").exists()


def test_timeout_and_output_budget(tmp_path):
    path = manifest(tmp_path, "import time; time.sleep(20)", timeout_seconds=1)
    result = run_experiment(tmp_path, path, backend="local")
    assert result["timed_out"] and not result["verification"]["verified"]
    path = manifest(tmp_path, "print('x' * 100000)")
    result = run_experiment(tmp_path, path, backend="local")
    assert result["output_limit_exceeded"]
    assert len(result["output"]) <= 64000
    assert not result["verification"]["verified"]


@pytest.mark.parametrize("value", [True, "1.0", None, float("nan")])
def test_invalid_metrics_rejected(tmp_path, value):
    (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": value}))
    spec = Experiment(command=["python"], artifact="metrics.json", metric="accuracy", minimum=.8)
    assert not verify_artifact(tmp_path, spec)["verified"]


def test_manifest_path_traversal_rejected():
    with pytest.raises(ValueError):
        Experiment(command=["python"], artifact="../metrics.json", metric="accuracy", minimum=.8)


def test_fixture_really_executes():
    root = Path(__file__).resolve().parents[1] / "examples/tiny-nlp"
    result = run_experiment(root, root / "experiment.json", backend="local")
    assert result["verification"]["value"] == 1.0
    assert len(result["verification"]["artifact"]["predictions"]) == 6


def test_forged_metric_is_rejected_by_independent_evaluator(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": 1.0, "predictions": [
        {"id": "a", "prediction": 0}, {"id": "b", "prediction": 1}]}))
    spec = Experiment(command=["python"], artifact="metrics.json", metric="accuracy", minimum=.8)
    result = verify_artifact(tmp_path, spec, truth=[{"id": "a", "label": 1}, {"id": "b", "label": 0}])
    assert result["status"] == "metric_mismatch" and not result["verified"]
    assert result["independent"]["wrong_ids"] == ["a", "b"]


def test_duplicate_or_missing_prediction_ids_are_rejected(tmp_path):
    spec = Experiment(command=["python"], artifact="metrics.json", metric="accuracy", minimum=.8)
    for predictions in [[{"id": "a", "prediction": 1}] * 2, []]:
        (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": 1, "predictions": predictions}))
        assert not verify_artifact(tmp_path, spec, truth=[{"id": "a", "label": 1}])["verified"]


def test_container_image_cannot_inject_docker_options():
    for value in ["--privileged", "image name", "", "--volume=/etc:/etc"]:
        with pytest.raises(ValueError):
            Experiment(command=["python"], artifact="metrics.json", metric="accuracy", minimum=.8,
                       container_image=value)
