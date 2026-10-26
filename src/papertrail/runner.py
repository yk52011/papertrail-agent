"""Execute an explicitly selected manifest in a disposable copy, with bounded resources."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: list[str] = Field(min_length=1, max_length=30)
    timeout_seconds: int = Field(default=30, ge=1, le=120)
    artifact: str
    metric: str
    minimum: float = Field(ge=0, le=1)
    ground_truth: str | None = None
    container_image: str = "python:3.11-slim"

    @field_validator("container_image")
    @classmethod
    def valid_image(cls, value):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/:@-]{0,199}", value):
            raise ValueError("container_image must be an image reference, not command options")
        return value

    @field_validator("artifact", "ground_truth")
    @classmethod
    def safe_artifact(cls, value):
        if value is None:
            return value
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("artifact must be a relative path inside the experiment directory")
        return value


def stop(process):
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass


def verify_artifact(directory: Path, spec: Experiment, *, truth: list[dict] | None = None) -> dict:
    path = (directory / spec.artifact).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        return {"status": "artifact_missing", "verified": False}
    if path.stat().st_size > 100_000:
        return {"status": "artifact_too_large", "verified": False}
    try:
        payload = json.loads(path.read_text())
        value = payload[spec.metric]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Metric must be finite and numeric")
        independent = None
        if truth is not None:
            # Evaluate against labels loaded BEFORE experiment execution, outside its process.
            expected = {row["id"]: row["label"] for row in truth}
            predictions = payload["predictions"]
            ids = [row["id"] for row in predictions]
            if len(ids) != len(set(ids)) or set(ids) != set(expected) or not expected:
                raise ValueError("Predictions must cover each reference ID exactly once")
            guessed = {row["id"]: row["prediction"] for row in predictions}
            labels = set(expected.values())
            if any(type(v) is not type(expected[k]) or v not in labels for k, v in guessed.items()):
                raise ValueError("Prediction label type or value is invalid")
            accuracy = sum(guessed[k] == expected[k] for k in expected) / len(expected)
            f1_values = []
            for label in labels:
                tp = sum(expected[k] == guessed[k] == label for k in expected)
                fp = sum(guessed[k] == label and expected[k] != label for k in expected)
                fn = sum(expected[k] == label and guessed[k] != label for k in expected)
                f1_values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
            independent = {"accuracy": accuracy, "macro_f1": sum(f1_values) / len(f1_values),
                           "count": len(expected),
                           "correct_ids": [k for k in expected if guessed[k] == expected[k]],
                           "wrong_ids": [k for k in expected if guessed[k] != expected[k]]}
            if spec.metric not in {"accuracy", "macro_f1"}:
                raise ValueError("Independent evaluator supports accuracy and macro_f1")
            if abs(value - independent[spec.metric]) > 1e-6:
                return {"status": "metric_mismatch", "verified": False, "reported": value,
                        "independent": independent, "artifact": payload}
            value = independent[spec.metric]
        return {"status": "threshold_met" if value >= spec.minimum else "threshold_not_met",
                "verified": value >= spec.minimum, "metric": spec.metric,
                "value": value, "minimum": spec.minimum, "artifact": payload,
                "independent": independent}
    except (ValueError, KeyError, TypeError):
        return {"status": "artifact_invalid", "verified": False}


def run_experiment(root: Path, manifest: Path, *, backend: str = "docker",
                   time_budget: int | None = None) -> dict:
    root = root.resolve()
    manifest = manifest.resolve()
    if not manifest.is_relative_to(root):
        raise ValueError("Manifest must be inside the repository")
    spec = Experiment.model_validate_json(manifest.read_text())
    if time_budget is not None:
        spec.timeout_seconds = min(spec.timeout_seconds, max(1, time_budget))
    truth, truth_digest = None, None
    if spec.ground_truth:
        truth_path = (root / spec.ground_truth).resolve()
        if not truth_path.is_relative_to(root) or not truth_path.is_file():
            raise ValueError("Ground-truth file must exist inside repository")
        raw = truth_path.read_bytes()
        if len(raw) > 500_000:
            raise ValueError("Ground truth exceeds 500 KB")
        truth = json.loads(raw)
        if (not isinstance(truth, list) or not truth or len(truth) > 5000
                or any(not isinstance(row, dict) or "id" not in row or "label" not in row
                       or not isinstance(row["id"], str) or type(row["label"]) not in {str, int}
                       for row in truth)):
            raise ValueError("Ground truth needs id and label rows")
        if len({r["id"] for r in truth}) != len(truth):
            raise ValueError("Duplicate ground-truth IDs")
        truth_digest = hashlib.sha256(raw).hexdigest()
    if backend not in {"docker", "local"}:
        raise ValueError("Unknown execution backend")
    if backend == "docker" and not shutil.which("docker"):
        raise ValueError("Docker unavailable. Install Docker or use local only for trusted code.")
    # Reject symlinks and oversized trees; do not copy virtual environments or hidden material.
    from .retrieval import SKIP
    total, count = 0, 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")]
        for name in dirs + names:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("Execution snapshot does not accept symbolic links")
        for name in names:
            if name.startswith("."):
                continue
            total += (Path(directory) / name).stat().st_size
            count += 1
    if total > 50_000_000 or count > 5000:
        raise ValueError("Execution snapshot exceeds 50 MB / 5000 files")
    container_name = "papertrail-" + uuid4().hex
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="papertrail-") as temporary:
        work = Path(temporary) / "workspace"
        shutil.copytree(root, work, ignore=lambda _, names:
                        [n for n in names if n in SKIP or n.startswith(".")])
        # An existing metric file must not count as the output of this run.
        artifact = work / spec.artifact
        if artifact.is_file():
            artifact.unlink()
        command = list(spec.command)
        if backend == "local" and command[0] in {"python", "python3"}:
            command[0] = sys.executable
        if backend == "docker":
            command = ["docker", "run", "--rm", "--pull=never", "--name", container_name,
                       "--network=none", "--memory=512m", "--cpus=1", "--pids-limit=64",
                       "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                       "--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", "/tmp:rw,size=64m",
                       "--mount", f"type=bind,source={work},target=/workspace",
                       "--workdir", "/workspace", spec.container_image, *spec.command]
        env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "LANG") if key in os.environ}
        env.update({"HOME": temporary, "PYTHONIOENCODING": "utf-8"})
        env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        output = bytearray()
        exceeded = threading.Event()
        process = subprocess.Popen(command, cwd=work, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)

        def collect():
            while True:
                block = process.stdout.read(4096)
                if not block:
                    break
                remaining = 64_000 - len(output)
                output.extend(block[:remaining])
                if len(block) > remaining:
                    exceeded.set()
                    stop(process)
                    break

        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        timed_out = False
        try:
            process.wait(timeout=spec.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop(process)
            process.wait()
        finally:
            # Also kill descendants that inherited the output pipe.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if backend == "docker":
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True,
                               timeout=15, check=False)
            reader.join(timeout=2)
            process.stdout.close()
        success = process.returncode == 0 and not timed_out and not exceeded.is_set()
        verification = verify_artifact(work, spec, truth=truth) if success else \
            {"status": "execution_failed", "verified": False}
        return {"backend": backend, "command": spec.command, "exit_code": process.returncode,
                "timed_out": timed_out, "output_limit_exceeded": exceeded.is_set(),
                "seconds": round(time.monotonic() - started, 3),
                "output": output.decode("utf-8", errors="replace"),
                "verification": verification, "paper_metric_status": "not_verified",
                "ground_truth_sha256": truth_digest,
                "scope": "Manifest output check only; not independent scientific validation."}
