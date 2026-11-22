#!/usr/bin/env python3
"""GitHub Actions entry point for the add-reproducer workflow.

`harness/pipeline.py` is the pipeline, and it deliberately knows nothing
about GitHub Actions: it *writes* the comment it composes to a file rather
than posting it anywhere. This script is the layer around it, split into
one subcommand per workflow job so that no job holds both a secret and the
issue's commands:

- `plan` fetches the issue and runs every stage up to the runner, including
  every model call. It needs `GITHUB_TOKEN` (read) and `OPENROUTER_API_KEY`,
  and it executes nothing taken from the issue. If there is anything to run,
  it writes `<out-dir>/<issue>-plan.json`.
- `execute` runs that plan, classifies the output and composes the comment.
  It refuses to start if either secret is in its environment.
- `post` posts the composed comment, if there is one, to the issue.

Everything that decides *whether* an issue deserves a comment lives in the
harness; this script adds no judgement of its own.

Usage (from `.github/add-reproducer/harness/`, so that the harness modules
and their one dependency are importable):

    uv run python ../run.py plan --repo canonical/operator --issue 2639 --out-dir "$PWD/out"
    env -u OPENROUTER_API_KEY -u GITHUB_TOKEN \\
        uv run python ../run.py execute --plan out/2639-plan.json --out-dir "$PWD/out"
    uv run python ../run.py post --repo canonical/operator --issue 2639 --comment out/2639.md

`--out-dir` must be absolute for `execute`: the harness embeds the charm
directory it builds underneath it into `juju deploy` command strings, and
juju 4 rejects a relative local-charm path that is not `./`-prefixed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "harness"))

import extraction_record  # noqa: E402
from models import Issue  # noqa: E402
from pipeline import Pipeline, PipelineResult, RunPlan, build_pipeline  # noqa: E402
from seams.llm import NoLLM  # noqa: E402
from seams.runner import SubprocessRunnerSeam  # noqa: E402

API_ROOT = "https://api.github.com"

# What `execute` refuses to start with.
SECRET_VARIABLES = ("OPENROUTER_API_KEY", "GITHUB_TOKEN")

#: The idempotency marker `harness/composer.py` appends to every comment it
#: composes is `<!-- add-reproducer:issue=<n>:run=<id> -->`. The run id
#: differs every time, so "have we already commented on this issue?" is a
#: match on everything up to it.
MARKER_PREFIX = "<!-- add-reproducer:issue={number}:"


def _request(url: str, token: str, *, method: str = "GET", payload: dict | None = None) -> object:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    request.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def _paginate(url: str, token: str) -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        batch = _request(f"{url}?per_page=100&page={page}", token)
        assert isinstance(batch, list)
        out.extend(batch)
        if len(batch) < 100:
            return out
        page += 1


def to_issue(raw: dict, comments: list[dict], repo: str) -> Issue:
    """Build the harness's `Issue` from a REST API issue and its comments.

    The harness's `Issue.from_dict` speaks `gh issue list --json ...`, which
    is camelCase and nests the author differently from the REST API, so the
    field names are mapped here rather than anywhere the harness can see
    them.
    """
    if "pull_request" in raw:
        raise SystemExit(f"{repo}#{raw.get('number')} is a pull request, not an issue")
    return Issue.from_dict(
        {
            "number": raw["number"],
            "title": raw.get("title") or "",
            "body": raw.get("body") or "",
            # A list of `{"name": ...}` objects either way.
            "labels": raw.get("labels") or [],
            # The REST API says "open"/"closed"; the harness's corpus and
            # `gh` both say "OPEN"/"CLOSED".
            "state": (raw.get("state") or "").upper(),
            "createdAt": raw.get("created_at") or "",
            "author": (raw.get("user") or {}).get("login", ""),
            "repo": repo,
            "comments": [
                {
                    "author": (comment.get("user") or {}).get("login", ""),
                    "body": comment.get("body") or "",
                    "createdAt": comment.get("created_at") or "",
                }
                for comment in comments
            ],
            # The issue *type* (GitHub's own field, not a label) is one of
            # the filter's drop rules.
            "type": raw.get("type"),
        }
    )


def fetch_issue(repo: str, number: int, token: str) -> Issue:
    raw = _request(f"{API_ROOT}/repos/{repo}/issues/{number}", token)
    assert isinstance(raw, dict)
    comments: list[dict] = []
    if raw.get("comments"):
        comments = _paginate(f"{API_ROOT}/repos/{repo}/issues/{number}/comments", token)
    return to_issue(raw, comments, repo)


def already_commented(repo: str, number: int, token: str) -> bool:
    marker = MARKER_PREFIX.format(number=number)
    comments = _paginate(f"{API_ROOT}/repos/{repo}/issues/{number}/comments", token)
    return any(marker in (comment.get("body") or "") for comment in comments)


def post_comment(repo: str, number: int, token: str, body: str) -> str:
    url = f"{API_ROOT}/repos/{repo}/issues/{number}/comments"
    created = _request(url, token, method="POST", payload={"body": body})
    assert isinstance(created, dict)
    return created.get("html_url", "")




def summarise(lines: list[str]) -> None:
    """Append to the job summary, when there is one to append to."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def set_output(name: str, value: str) -> None:
    """Set a step output, when there is a step to set it on."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def write_extraction_record(record: dict | None, number: int, out_dir: Path) -> list[str]:
    """Write the extraction record, print it to the log, and return its lines.

    A run that stops at the in-scope gate has nothing else to record, and
    that is the stop whose reasons most need recording.
    `-extraction.json` matches the workflow's `out/*.json` upload glob, so
    the artefact exists for a gate stop too."""
    if record is None:
        return []
    (out_dir / f"{number}-extraction.json").write_text(json.dumps(record, indent=2, default=str))
    lines = extraction_record.render(record)
    for line in lines:
        print(line)
    return lines


def report(repo: str, number: int, result: PipelineResult, record_lines: list[str], out_dir: Path) -> None:
    """Log a finished pipeline result, write its run record and comment to
    `out_dir`, and add it to the job summary."""
    print(f"#{result.issue_number}: stage={result.stage_reached} outcome={result.outcome} reason={result.reason}")
    if result.run_result is not None:
        (out_dir / f"{number}-run.json").write_text(
            json.dumps(
                {
                    "branch": result.branch,
                    "hypothesis": asdict(result.hypothesis) if result.hypothesis is not None else None,
                    "surface": asdict(result.surface) if result.surface is not None else None,
                    **asdict(result.run_result),
                },
                indent=2,
                default=str,
            )
        )
    summary = [
        f"### {repo}#{number}",
        "",
        f"- stage reached: `{result.stage_reached}`",
        f"- outcome: `{result.outcome}`",
        f"- reason: {result.reason or '-'}",
    ]
    if record_lines:
        summary += ["", "```", *record_lines, "```"]
    if not result.comment:
        summary.append("- comment: **none composed** (the pipeline stayed silent)")
        summarise(summary)
        return
    comment_path = out_dir / f"{number}.md"
    comment_path.write_text(result.comment)
    print(f"  composed comment written to {comment_path}")
    summary += [
        "- comment: composed",
        "",
        "<details><summary>The composed comment</summary>",
        "",
        result.comment,
        "",
        "</details>",
    ]
    summarise(summary)


def plan(args: argparse.Namespace) -> int:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise SystemExit("GITHUB_TOKEN is not set")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_if_commented and already_commented(args.repo, args.issue, token):
        print(f"{args.repo}#{args.issue}: a reproducer comment is already there; nothing to do")
        summarise([f"### {args.repo}#{args.issue}", "", "Already commented on; skipped."])
        return 0

    issue = fetch_issue(args.repo, args.issue, token)
    pipeline = build_pipeline(fixture_mode=False, work_dir=out_dir / "work")
    # The workflow run is the run id, so the marker on a posted comment names
    # the run that produced it.
    planned = pipeline.plan_for_issue(issue, run_id=os.environ.get("GITHUB_RUN_ID"))
    record_lines = write_extraction_record(planned.extraction_record, issue.number, out_dir)
    if isinstance(planned, PipelineResult):
        report(args.repo, args.issue, planned, record_lines, out_dir)
        return 0

    plan_path = out_dir / f"{issue.number}-plan.json"
    plan_path.write_text(json.dumps(planned.to_dict(), indent=2, default=str))
    print(f"#{issue.number}: planned a {planned.branch} run, written to {plan_path}")
    summary = [f"### {args.repo}#{args.issue}", "", f"- planned: a `{planned.branch}` run"]
    if record_lines:
        summary += ["", "```", *record_lines, "```"]
    summarise(summary)
    set_output("planned", "true")
    return 0


def execute(args: argparse.Namespace) -> int:
    # This half runs commands taken from the issue, so anything in its
    # environment is readable by whoever wrote the issue.
    present = [name for name in SECRET_VARIABLES if os.environ.get(name)]
    if present:
        raise SystemExit(f"refusing to run issue-derived commands with {', '.join(present)} set")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_plan = RunPlan.from_dict(json.loads(args.plan.read_text()))
    pipeline = Pipeline(NoLLM(), SubprocessRunnerSeam(), work_dir=out_dir / "work")
    result = pipeline.execute_plan(run_plan)
    record_lines = extraction_record.render(run_plan.extraction_record) if run_plan.extraction_record else []
    report(run_plan.issue.repo, run_plan.issue.number, result, record_lines, out_dir)
    if result.comment:
        set_output("composed", "true")
    return 0


def post(args: argparse.Namespace) -> int:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise SystemExit("GITHUB_TOKEN is not set")
    heading = [f"### {args.repo}#{args.issue}", ""]
    if not args.comment.exists():
        print(f"{args.repo}#{args.issue}: no comment was composed; nothing to post")
        summarise([*heading, "No comment was composed; nothing posted."])
        return 0
    if already_commented(args.repo, args.issue, token):
        print(f"{args.repo}#{args.issue}: a reproducer comment is already there; nothing to do")
        summarise([*heading, "Already commented on; skipped."])
        return 0
    try:
        url = post_comment(args.repo, args.issue, token, args.comment.read_text())
    except urllib.error.HTTPError as exc:
        summarise([*heading, f"Posting failed ({exc.code} {exc.reason})."])
        raise
    print(f"  posted: {url}")
    summarise([*heading, f"[Posted]({url})."])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    plan_parser = commands.add_parser("plan", help="fetch the issue, make every model call, and write a run plan")
    plan_parser.add_argument("--repo", required=True, help="owner/name the issue lives in")
    plan_parser.add_argument("--issue", type=int, required=True, help="issue number")
    plan_parser.add_argument("--out-dir", type=Path, required=True, help="where the plan and records go")
    plan_parser.add_argument(
        "--skip-if-commented",
        action="store_true",
        help="stop before any model call if the issue already has a reproducer comment",
    )
    plan_parser.set_defaults(func=plan)

    execute_parser = commands.add_parser("execute", help="run a plan, classify it, and compose the comment")
    execute_parser.add_argument("--plan", type=Path, required=True, help="the plan `plan` wrote")
    execute_parser.add_argument(
        "--out-dir", type=Path, required=True, help="absolute path for the run record and comment"
    )
    execute_parser.set_defaults(func=execute)

    post_parser = commands.add_parser("post", help="post a composed comment to the issue")
    post_parser.add_argument("--repo", required=True, help="owner/name the issue lives in")
    post_parser.add_argument("--issue", type=int, required=True, help="issue number")
    post_parser.add_argument("--comment", type=Path, required=True, help="the comment `execute` wrote, if any")
    post_parser.set_defaults(func=post)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
