"""Are `commands[]` actually runnable? (PLAN.md Approach §4 pre-run gate.)

`spike-step-5/live-llm/RESULT.md` Finding 2: the extraction schema says
`"commands": [str]` and nothing about what a command *is*, so a live model
emits prose ("Run a charm test that checks the default relation settings"),
bare Python expressions (`relation = self.model.get_relation(...)`), or
pytest test names — none of which are shell. Post-fix live run: 0 of 7
issues produced a runnable command sequence.

Handing those to a shell is not merely useless, it is actively harmful.
`bash -c 'Run a charm test ...'` exits **127** (`Run: command not found`),
and the classifier's rung 6 turns any non-zero last command into
`reproduced (weaker)`, which is in `COMMENT_OUTCOMES` — so prose becomes a
maintainer-visible comment claiming the bug reproduced. Demonstrated on
the real #2185/#2341/#2327 live extractions. That is the false-comment
mode PLAN.md Goal §4 ("nothing useful produced → stay silent") and Steps
§1's bias-toward-false-drop exist to prevent.

So this module answers one question, conservatively: would these strings
survive contact with a shell at all? It deliberately does not judge
whether they reproduce the bug — that's the classifier's job, downstream,
and it only gets to run if the answer here is yes.

The check is a property, not a vibe: a command is shell-shaped when it
parses as shell **and** its first token names something the runner can
actually execute. Nothing here tries to detect English.
"""

from __future__ import annotations

import dataclasses
import enum
import re
import shutil
import subprocess

# Tools the runner is documented to provide: the host scratch directory for
# `substrate: none` (PLAN.md Approach §4) plus the multipass VM's
# concierge/juju/charmcraft for `lxd|k8s`. Kept as an explicit allowlist
# rather than probing PATH, so the verdict is identical on this box, in the
# VM, and on a GHA runner — a `which`-only check would call `juju` prose
# when run locally.
RUNNER_TOOLS = frozenset(
    {
        # python / packaging
        "python", "python3", "uv", "uvx", "pip", "pip3", "pytest", "tox",
        # vcs + fetch
        "git", "gh", "curl", "wget",
        # juju / charm substrate
        "juju", "charmcraft", "concierge", "kubectl", "microk8s", "k8s", "lxc", "lxd", "snap",
        # ordinary shell utilities a repro recipe legitimately uses
        "cat", "cp", "mv", "rm", "mkdir", "rmdir", "ls", "chmod", "chown", "ln",
        "echo", "printf", "tee", "sed", "awk", "grep", "sort", "head", "tail", "wc",
        "touch", "diff", "patch", "tar", "unzip", "env", "sleep", "sudo", "bash", "sh",
        "make", "docker", "which", "test", "find", "xargs", "jq", "yq",
    }
)

# Shell builtins and control-flow keywords, which `which` will never find.
SHELL_BUILTINS = frozenset(
    {
        "cd", "export", "set", "unset", "source", ".", "eval", "exec", "exit",
        "true", "false", "read", "shift", "trap", "wait", "local", "return",
        "pushd", "popd", "alias", "umask", "if", "then", "else", "elif", "fi",
        "for", "while", "until", "do", "done", "case", "esac", "function", "time",
    }
)

# `FOO=bar cmd ...` and bare `FOO=bar` are both valid shell.
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


class Shape(str, enum.Enum):
    """What a single `commands[]` entry turns out to be."""

    SHELL = "shell"  # parses as shell, first token is executable
    PROSE = "prose"  # parses, but names something unrunnable ("Run a charm test ...")
    CODE_FRAGMENT = "code_fragment"  # does not parse as shell at all
    COMMENT = "comment"  # `# write test_cwd.py: ...` -- a placeholder, not a command
    EMPTY = "empty"


@dataclasses.dataclass
class RunnabilityReport:
    shapes: list[Shape]
    runnable: bool
    reason: str

    @property
    def unrunnable_commands(self) -> list[int]:
        """Indices of entries that would fail on contact with a shell."""
        return [i for i, s in enumerate(self.shapes) if s in (Shape.PROSE, Shape.CODE_FRAGMENT)]


def _parses_as_shell(command: str) -> bool:
    """True if `bash -n` accepts it. Catches Python fragments like
    `relation.data[self.app]` (syntax error near unexpected token `[`).

    If bash isn't installed we can't tell, so we say yes and let the
    first-token check carry the decision -- better than declaring every
    command un-runnable on a box without bash.
    """
    bash = shutil.which("bash")
    if bash is None:
        return True
    try:
        proc = subprocess.run(
            [bash, "-n", "-c", command], capture_output=True, text=True, timeout=10
        )
    except (subprocess.SubprocessError, OSError):
        return True
    return proc.returncode == 0


def _first_token(command: str) -> str:
    """The word that has to name an executable, skipping `VAR=val` prefixes
    and any `sudo`/`env`/`time` wrapper."""
    tokens = command.strip().split()
    for token in tokens:
        if _ASSIGNMENT_RE.match(token):
            continue
        if token in ("sudo", "env", "time", "command", "exec"):
            continue
        return token
    return ""


def classify_command(command: str, *, extra_tools: frozenset[str] = frozenset()) -> Shape:
    text = command.strip()
    if not text:
        return Shape.EMPTY
    if text.startswith("#"):
        return Shape.COMMENT
    if not _parses_as_shell(text):
        return Shape.CODE_FRAGMENT
    token = _first_token(text)
    if not token:
        return Shape.EMPTY
    # A path, a pipeline/redirect/subshell opener, or an assignment-only line
    # is shell by construction.
    if token.startswith(("./", "/", "(", "{", "$")) or _ASSIGNMENT_RE.match(token):
        return Shape.SHELL
    base = token.rsplit("/", 1)[-1]
    if base in SHELL_BUILTINS or base in RUNNER_TOOLS or base in extra_tools:
        return Shape.SHELL
    # Heredoc bodies and multi-line scripts: judge by the first line only if
    # any line looks like a real command, since `cat > f << 'EOF' ... EOF`
    # arrives as one entry.
    if "\n" in text:
        for line in text.splitlines():
            if classify_command(line, extra_tools=extra_tools) is Shape.SHELL:
                return Shape.SHELL
    return Shape.PROSE


def assess(commands: list[str], *, extra_tools: frozenset[str] = frozenset()) -> RunnabilityReport:
    """Approach §4's pre-run gate.

    Runnable requires **both**: at least one genuinely executable command,
    and no entry that would blow up on contact with a shell. A prose line
    anywhere in the sequence means the recipe is broken -- it isn't a
    harmless annotation, it's an exit-127 that the classifier reads as
    evidence the bug reproduced.
    """
    shapes = [classify_command(c, extra_tools=extra_tools) for c in commands]
    if not commands:
        return RunnabilityReport(shapes, False, "commands[] is empty")
    broken = [(i, s) for i, s in enumerate(shapes) if s in (Shape.PROSE, Shape.CODE_FRAGMENT)]
    if broken:
        i, shape = broken[0]
        return RunnabilityReport(
            shapes,
            False,
            f"commands[{i}] is {shape.value}, not a runnable command: {commands[i].strip()[:80]!r}",
        )
    if not any(s is Shape.SHELL for s in shapes):
        kinds = ", ".join(sorted({s.value for s in shapes})) or "nothing"
        return RunnabilityReport(shapes, False, f"no executable command in commands[] (only {kinds})")
    return RunnabilityReport(shapes, True, "all commands are shell-shaped")


def runnable_fraction(command_lists: list[list[str]]) -> tuple[int, int]:
    """`(runnable, total)` over several hypotheses -- Approach §3's
    `commands_runnable` metric, which the spikes measured by hand.
    """
    total = len(command_lists)
    return sum(1 for cs in command_lists if assess(cs).runnable), total
