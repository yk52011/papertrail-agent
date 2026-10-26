"""CLI: offline evidence first; optional API agent; explicit experiment execution."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .audit import audit_repository, markdown_report
from .retrieval import RepositoryIndex
from .store import save_run


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(prog="papertrail")
    commands = parser.add_subparsers(dest="command", required=True)
    workflow = commands.add_parser("reproduce", help="Run the persistent execute/recover workflow")
    workflow.add_argument("repo", type=Path)
    workflow.add_argument("--task", default="task.json")
    workflow.add_argument("--policy", choices=["none", "retry", "rules", "llm"], default="rules")
    workflow.add_argument("--backend", choices=["docker", "local"], default="docker")
    workflow.add_argument("--max-attempts", type=int, default=3)
    workflow.add_argument("--max-seconds", type=int, default=90)
    workflow.add_argument("--step", action="store_true", help="Pause before each recovery patch")
    resume = commands.add_parser("resume", help="Continue a checkpointed task")
    resume.add_argument("job_id")
    resume.add_argument("--cancel", action="store_true")
    commands.add_parser("jobs", help="List persistent tasks")
    for name in ("audit", "search", "ask", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("repo", type=Path)
        if name in {"search", "ask"}:
            sub.add_argument("question")
            sub.add_argument("--semantic", action="store_true",
                             help="Send indexed text to configured embedding API")
        if name in {"audit", "ask"}:
            sub.add_argument("--output", type=Path, default=Path("outputs/report.md"))
        if name == "ask":
            sub.add_argument("--max-turns", type=int, default=4)
        if name == "run":
            sub.add_argument("--manifest", type=Path, required=True,
                             help="Path relative to repository")
            sub.add_argument("--backend", choices=["docker", "local"], default="docker")
    args = parser.parse_args()
    try:
        if args.command in {"reproduce", "resume", "jobs"}:
            from .jobs import JobStore
            from .workflow import job_summary, resume_job, start_job
            if args.command == "jobs":
                print(json.dumps(JobStore().list(), ensure_ascii=False, indent=2))
                return
            if args.command == "reproduce":
                state = start_job(args.repo, task_file=args.task, backend=args.backend,
                                  policy=args.policy, max_attempts=args.max_attempts,
                                  max_seconds=args.max_seconds, step_mode=args.step)
            else:
                state = resume_job(args.job_id, proceed=not args.cancel)
            print(json.dumps({"job_id": state["job_id"], "status": state["status"],
                              "summary": job_summary(state), "events": state["events"]},
                             ensure_ascii=False, indent=2))
            if state["status"] not in {"succeeded", "paused"}:
                raise SystemExit(2)
            return
        if args.command == "run":
            from .runner import run_experiment
            result = run_experiment(args.repo, args.repo / args.manifest, backend=args.backend)
        else:
            embeddings = None
            if getattr(args, "semantic", False):
                from .agent import configured_embeddings
                embeddings = configured_embeddings()
                if embeddings is None:
                    raise ValueError("Set PAPERTRAIL_EMBEDDING_MODEL for --semantic")
            index = RepositoryIndex(args.repo, embeddings=embeddings)
            if args.command == "search":
                result = index.search(args.question, mode="hybrid" if embeddings else "bm25")
            else:
                audit = audit_repository(index)
                result = audit
                question, answer = "", ""
                if args.command == "ask":
                    from .agent import ask
                    result = ask(index, args.question, max_turns=args.max_turns)
                    question, answer = args.question, result["answer"]
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(markdown_report(audit, question=question, answer=answer),
                                       encoding="utf-8")
        run_id = save_run({"operation": args.command, "result": result})
        print(json.dumps({"run_id": run_id, "result": result}, ensure_ascii=False, indent=2))
        if args.command == "run" and not result["verification"]["verified"]:
            raise SystemExit(2)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"papertrail: {exc}\n")


if __name__ == "__main__":
    main()
