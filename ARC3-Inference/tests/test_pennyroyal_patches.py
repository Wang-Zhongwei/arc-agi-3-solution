"""Source patches the Pennyroyal launcher applies to the installed SGLang."""

import os
import py_compile
import shutil
from collections import defaultdict
from pathlib import Path

import pytest

from inference.framework import pennyroyal_patches as patches


def write_anchor_tree(root: Path, skip: str | None = None) -> dict[str, str]:
    """A stand-in site-packages whose files hold each edit's anchor once."""
    files = defaultdict(str)
    for rel, anchor, _ in patches.EDITS:
        if anchor != skip:
            files[rel] += f"# before\n{anchor}# after\n"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return dict(files)


def test_apply_edit_inserts_once_and_is_idempotent():
    text = "a\nANCHOR\nb\n"
    once = patches._apply_edit(text, "ANCHOR\n", "NEW\nANCHOR\n", "f")
    assert once == "a\nNEW\nANCHOR\nb\n"
    assert patches._apply_edit(once, "ANCHOR\n", "NEW\nANCHOR\n", "f") == once


def test_apply_edit_rejects_missing_or_repeated_anchor():
    with pytest.raises(RuntimeError, match="found 0"):
        patches._apply_edit("a\n", "ANCHOR\n", "NEW\n", "f")
    with pytest.raises(RuntimeError, match="found 2"):
        patches._apply_edit("ANCHOR\nANCHOR\n", "ANCHOR\n", "NEW\n", "f")


def test_apply_patches_every_file_then_nothing(tmp_path):
    written = write_anchor_tree(tmp_path)
    changed = patches.apply(tmp_path)
    assert set(changed) == set(written) | set(patches.overlay_files())
    for rel, anchor, replacement in patches.EDITS:
        assert replacement in (tmp_path / rel).read_text()
    assert patches.apply(tmp_path) == []


def test_missing_anchor_leaves_every_file_untouched(tmp_path):
    marlin_anchor = next(a for rel, a, _ in patches.EDITS if rel == patches.MARLIN_FP4)
    written = write_anchor_tree(tmp_path, skip=marlin_anchor)
    with pytest.raises(RuntimeError, match="marlin_utils_fp4"):
        patches.apply(tmp_path)
    for rel, text in written.items():
        assert (tmp_path / rel).read_text() == text
    assert not any((tmp_path / rel).exists() for rel in patches.overlay_files())


def test_overlay_ships_the_fp8_kernel_module():
    files = patches.overlay_files()
    assert "sglang/kernels/ops/gemm/sm120_lowm_fp8_gemm.py" in files
    for source in files.values():
        py_compile.compile(str(source), doraise=True)


@pytest.mark.skipif(
    not os.environ.get("PENNYROYAL_SGLANG_SITE"),
    reason="set PENNYROYAL_SGLANG_SITE to a site-packages dir holding the pinned Pennyroyal sglang",
)
def test_patches_apply_to_pinned_pennyroyal(tmp_path):
    shutil.copytree(Path(os.environ["PENNYROYAL_SGLANG_SITE"]) / "sglang", tmp_path / "sglang")
    changed = patches.apply(tmp_path)
    assert len(changed) == len({rel for rel, _, _ in patches.EDITS}) + len(patches.overlay_files())
    for rel in changed:
        py_compile.compile(str(tmp_path / rel), doraise=True)
    assert patches.apply(tmp_path) == []
