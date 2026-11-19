"""Install the bundled pyenrichedheatmap Agent Skill for Claude Code.

This module backs the ``pyenrichedheatmap-install-skills`` console script. It copies
the skill that ships inside the installed package into a directory where
Claude Code discovers skills, giving a coding agent pyenrichedheatmap-specific
guidance (API contracts, coordinate conventions, and rendering pitfalls) in every
project.

Claude Code does not scan ``site-packages`` for skills, so the skill has to
be copied once. The default destination is the personal skill root,
``~/.claude/skills/pyenrichedheatmap``, making the skill available across all
projects. Pass ``--dest .claude/skills/pyenrichedheatmap`` to install into a
single project instead. Copying is opt-in and never happens on import or
``pip install``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SKILL_NAME = "pyenrichedheatmap"


def bundled_skill_dir() -> Path:
    """Directory of the skill shipped with this package (contains ``SKILL.md``)."""
    return Path(__file__).resolve().parent / "data"


def default_dest() -> Path:
    """Return the default personal Claude Code skill path (``~/.claude/skills/pyenrichedheatmap``)."""
    return Path.home() / ".claude" / "skills" / SKILL_NAME


def install_skill(dest: Path | None = None, force: bool = False) -> Path:
    """Copy the bundled skill to ``dest`` (default :func:`default_dest`).

    An existing ``dest`` is only replaced when ``force=True`` so that an
    upgrade reflects the installed ``pyenrichedheatmap`` version cleanly.
    """
    src = bundled_skill_dir()
    if not (src / "SKILL.md").is_file():
        raise FileNotFoundError(
            f"Bundled skill not found at {src}; the package may be installed without its data."
        )
    target = default_dest() if dest is None else Path(dest).expanduser()
    if target.exists():
        if not force:
            raise FileExistsError(f"{target} already exists; re-run with --force to overwrite")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        target,
        ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints", ".DS_Store"),
    )
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``pyenrichedheatmap-install-skills``."""
    parser = argparse.ArgumentParser(
        prog="pyenrichedheatmap-install-skills",
        description="Install the pyenrichedheatmap Agent Skill for Claude Code.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="destination directory (default: ~/.claude/skills/pyenrichedheatmap)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing installation",
    )
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="print the bundled skill directory and exit without installing",
    )
    args = parser.parse_args(argv)

    if args.print_path:
        print(bundled_skill_dir())
        return 0
    try:
        dest = install_skill(args.dest, args.force)
    except (FileExistsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Installed pyenrichedheatmap skill to {dest}")
    print("It will be available to Claude Code in your next session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
