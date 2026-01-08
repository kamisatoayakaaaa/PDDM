import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path


def get_git_info():
    commit = None
    branch = None
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        pass
    try:
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        pass
    return commit, branch


def format_dict_md(d, indent=0):
    lines = []
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}- {k}:")
            lines.extend(format_dict_md(v, indent + 1))
        else:
            lines.append(f"{pad}- {k}: {v}")
    return lines


def log_experiment(
    run_name,
    config,
    metrics,
    log_path="experiments.md",
    extra_text=None,
):
    root = Path(__file__).resolve().parents[1]
    log_path = Path(log_path)
    if not log_path.is_absolute():
        log_path = root / log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmd = " ".join(["python"] + sys.argv[0:])

    commit, branch = get_git_info()

    lines = []
    if not log_path.exists():
        lines.append("# Experiments Log\n")

    lines.append(f"## {now} · {run_name}\n")
    lines.append(f"- Working dir: `{os.getcwd()}`")
    lines.append(f"- Command: `{cmd}`")
    if branch:
        lines.append(f"- Git branch: `{branch}`")
    if commit:
        lines.append(f"- Git commit: `{commit}`")

    if config:
        lines.append("")
        lines.append("### Config")
        lines.extend(format_dict_md(config))

    if metrics:
        lines.append("")
        lines.append("### Metrics")
        lines.extend(format_dict_md(metrics))

    if extra_text:
        lines.append("")
        lines.append("### Notes")
        lines.append(extra_text)

    lines.append("\n---\n\n")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
