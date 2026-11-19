"""Tests verifying the bundled Claude Code skill, routes, API names, installer, and check script."""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

import enriched_heatmap as eh
from enriched_heatmap._skills.install import bundled_skill_dir, install_skill, main

SKILL = bundled_skill_dir()


def test_skill_frontmatter_and_routes():
    text = (SKILL / "SKILL.md").read_text()
    front = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert front, "SKILL.md needs YAML frontmatter"
    fields = dict(line.split(": ", 1) for line in front.group(1).splitlines())
    assert fields["name"] == "pyenrichedheatmap"
    assert 100 < len(fields["description"]) <= 1024

    routed = set(re.findall(r"`((?:references|scripts)/[\w.-]+)`", text))
    assert routed, "router table is empty"
    for rel in routed:
        assert (SKILL / rel).is_file(), f"SKILL.md routes to missing {rel}"

    on_disk = {f"references/{p.name}" for p in (SKILL / "references").glob("*.md")}
    assert on_disk <= routed, f"reference files not routed from SKILL.md: {on_disk - routed}"


def test_references_only_mention_real_api():
    names: set[str] = set()
    for path in [SKILL / "SKILL.md", *(SKILL / "references").glob("*.md")]:
        content = path.read_text()
        names |= set(re.findall(r"\beh\.([A-Za-z_]\w*)", content))
        for ref in re.findall(r"`(references/[\w.-]+)`", content):
            assert (SKILL / ref).is_file(), f"{path.name} points to missing {ref}"
    missing = sorted(n for n in names if not hasattr(eh, n))
    assert not missing, f"skill mentions eh.{missing} which enriched_heatmap does not export"


def test_install_cli(tmp_path, capsys):
    dest = install_skill(tmp_path / "skills" / "pyenrichedheatmap")
    assert (dest / "SKILL.md").read_text() == (SKILL / "SKILL.md").read_text()
    assert (dest / "references" / "data-path.md").is_file()
    assert (dest / "scripts" / "check_regions.py").is_file()

    with pytest.raises(FileExistsError):
        install_skill(dest)
    install_skill(dest, force=True)

    assert main(["--dest", str(dest)]) == 1
    assert main(["--dest", str(dest), "--force"]) == 0
    assert main(["--print-path"]) == 0
    assert capsys.readouterr().out.strip().endswith(str(SKILL))


def test_check_regions_script(tmp_path):
    script = SKILL / "scripts" / "check_regions.py"
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t100\t300\tp1\t0\t+\nchr9\t500\t700\tp2\t0\t-\n")
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("chr1\t10000\n")
    out = subprocess.run([sys.executable, str(script), str(bed), "--genome", str(sizes)],
                         capture_output=True, text=True)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "chromosomes not in genome" in out.stdout and "chr9" in out.stdout
    assert "touch a chromosome edge" in out.stdout
    sizes.write_text("chr1\t10000\nchr9\t10000\n")
    ok = subprocess.run([sys.executable, str(script), str(bed), "--genome", str(sizes)],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
