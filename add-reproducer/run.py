#!/usr/bin/env python3
"""GitHub Actions entry point for the add-reproducer workflow.

`harness/pipeline.py` is the pipeline, and it deliberately knows nothing
about GitHub Actions: it reads a JSON dump of issues, and it *writes* the
comment it composes to a file rather than posting it anywhere. This script
is the thin layer either side of that -- fetch one issue from the REST API,
run it through the pipeline, and (only when explicitly asked) post the
composed comment back to the issue.

Everything that decides *whether* an issue deserves a comment lives in the
harness: the deterministic in-scope filter, the model's own scope second
opinion, the confidence gate, the runnability gate, and the outcome ladder,
which composes nothing at all for most outcomes. This script adds no
judgement of its own. Its only switch is `--post`, which decides whether an
already-composed comment is delivered or merely printed -- so a run with
`--post` off exercises the identical pipeline and shows exactly what would
have been said.

Usage (from `add-reproducer/harness/`, so that the harness modules and their
one dependency are importable):

    uv run python ../run.py --repo canonical/operator --issue 2639 \
        --out-dir "$PWD/out"

`--out-dir` must be absolute: the harness embeds the charm directory it
builds underneath it into `juju deploy` command strings, and juju 4 rejects
a relative local-charm path that is not `./`-prefixed.
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

from models import Issue  # noqa: E402
from pipeline import build_pipeline  # noqa: E402

API_ROOT = "https://api.github.com"

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="owner/name the issue lives in")
    parser.add_argument("--issue", type=int, required=True, help="issue number")
    parser.add_argument("--out-dir", type=Path, required=True, help="absolute path for the run record and comment")
    parser.add_argument(
        "--post",
        action="store_true",
        help=(
            "actually post the composed comment to the issue. Without it the "
            "pipeline runs identically and the comment it would have posted is "
            "written to --out-dir and the job summary instead."
        ),
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        parser.error("GITHUB_TOKEN is not set")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.post and already_commented(args.repo, args.issue, token):
        print(f"{args.repo}#{args.issue}: a reproducer comment is already there; nothing to do")
        summarise([f"### {args.repo}#{args.issue}", "", "Already commented on; skipped."])
        return 0

    issue = fetch_issue(args.repo, args.issue, token)
    pipeline = build_pipeline(fixture_mode=False, work_dir=out_dir / "work")
    # The workflow run is the run id, so the marker on a posted comment names
    # the job that produced it.
    result = pipeline.run_for_issue(issue, run_id=os.environ.get("GITHUB_RUN_ID"))

    print(f"#{result.issue_number}: stage={result.stage_reached} outcome={result.outcome} reason={result.reason}")
    if result.run_result is not None:
        (out_dir / f"{issue.number}-run.json").write_text(
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
        f"### {args.repo}#{args.issue}",
        "",
        f"- stage reached: `{result.stage_reached}`",
        f"- outcome: `{result.outcome}`",
        f"- reason: {result.reason or '-'}",
    ]
    if not result.comment:
        summary.append("- comment: **none composed** (the pipeline stayed silent)")
        summarise(summary)
        return 0

    comment_path = out_dir / f"{issue.number}.md"
    comment_path.write_text(result.comment)
    print(f"  composed comment written to {comment_path}")

    if not args.post:
        summary += [
            "- comment: composed, **not posted** (`post_comment` is off)",
            "",
            "<details><summary>The comment that would have been posted</summary>",
            "",
            result.comment,
            "",
            "</details>",
        ]
        summarise(summary)
        return 0

    try:
        url = post_comment(args.repo, args.issue, token, result.comment)
    except urllib.error.HTTPError as exc:
        summary.append(f"- comment: composed, but posting failed ({exc.code} {exc.reason})")
        summarise(summary)
        raise
    print(f"  posted: {url}")
    summary.append(f"- comment: [posted]({url})")
    summarise(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
