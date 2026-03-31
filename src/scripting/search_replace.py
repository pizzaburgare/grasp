#!/usr/bin/env python
"""Flexible search-and-replace utilities vendored from aider-chat.

Vendored from https://github.com/Aider-AI/aider (Apache-2.0).
aider-chat requires Python <3.13; we copy only this file and stub its
two aider-internal imports below.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

try:
    import git
    from git.exc import GitError, ODBError
except ImportError:
    git = None  # type: ignore[assignment]
    GitError = ODBError = Exception

from diff_match_patch import diff_match_patch

# ---------------------------------------------------------------------------
# Stubs for aider-internal helpers
# ---------------------------------------------------------------------------


def dump(*args: Any) -> None:
    """Debug printer (no-op in production)."""


class GitTemporaryDirectory:
    """Context manager that creates a temp git repo directory."""

    def __enter__(self) -> str:
        self._tmpdir = tempfile.TemporaryDirectory()
        dname = self._tmpdir.name
        if git is not None:
            repo = git.Repo.init(dname)
            repo.config_writer().set_value("user", "name", "aider").release()
            repo.config_writer().set_value("user", "email", "aider@example.com").release()
        return dname

    def __exit__(self, *args: Any) -> None:
        self._tmpdir.cleanup()


# ---------------------------------------------------------------------------
# RelativeIndenter
# ---------------------------------------------------------------------------


class RelativeIndenter:
    """Rewrites text files to have relative indentation, which involves
    reformatting the leading white space on lines.  This format makes
    it easier to search and apply edits to pairs of code blocks which
    may differ significantly in their overall level of indentation.

    It removes leading white space which is shared with the preceding
    line.
    """

    def __init__(self, texts: list[str]) -> None:
        """Based on the texts, choose a unicode character that isn't in any of them."""
        chars: set[str] = set()
        for text in texts:
            chars.update(text)

        arrow = "\u2190"  # ←
        if arrow not in chars:
            self.marker = arrow
        else:
            self.marker = self.select_unique_marker(chars)

    def select_unique_marker(self, chars: set[str]) -> str:
        """Find a Unicode character not present in the given character set."""
        for codepoint in range(0x10FFFF, 0x10000, -1):
            marker = chr(codepoint)
            if marker not in chars:
                return marker
        raise ValueError("Could not find a unique marker")

    def make_relative(self, text: str) -> str:
        """Transform text to use relative indents."""
        if self.marker in text:
            raise ValueError(f"Text already contains the outdent marker: {self.marker}")

        lines = text.splitlines(keepends=True)
        output = []
        prev_indent = ""
        for line in lines:
            line_without_end = line.rstrip("\n\r")
            len_indent = len(line_without_end) - len(line_without_end.lstrip())
            indent = line[:len_indent]
            change = len_indent - len(prev_indent)
            if change > 0:
                cur_indent = indent[-change:]
            elif change < 0:
                cur_indent = self.marker * -change
            else:
                cur_indent = ""

            out_line = cur_indent + "\n" + line[len_indent:]
            output.append(out_line)
            prev_indent = indent

        res = "".join(output)
        return res

    def make_absolute(self, text: str) -> str:
        """Transform text from relative back to absolute indents."""
        lines = text.splitlines(keepends=True)
        output = []
        prev_indent = ""
        for i in range(0, len(lines), 2):
            dent = lines[i].rstrip("\r\n")
            non_indent = lines[i + 1]

            if dent.startswith(self.marker):
                len_outdent = len(dent)
                cur_indent = prev_indent[:-len_outdent]
            else:
                cur_indent = prev_indent + dent

            out_line = non_indent if not non_indent.rstrip("\r\n") else cur_indent + non_indent

            output.append(out_line)
            prev_indent = cur_indent

        res = "".join(output)
        if self.marker in res:
            raise ValueError("Error transforming text back to absolute indents")

        return res


# ---------------------------------------------------------------------------
# DMP helpers
# ---------------------------------------------------------------------------


def map_patches(texts: list[str], patches: list, debug: bool) -> list:
    """Remap patch positions from search text coordinates to original text coordinates."""
    search_text, _, original_text = texts

    dmp = diff_match_patch()
    dmp.Diff_Timeout = 5

    diff_s_o = dmp.diff_main(search_text, original_text)

    if debug:
        html = dmp.diff_prettyHtml(diff_s_o)
        Path("tmp.html").write_text(html)
        dump(len(search_text))
        dump(len(original_text))

    for patch in patches:
        start1 = patch.start1
        start2 = patch.start2

        patch.start1 = dmp.diff_xIndex(diff_s_o, start1)
        patch.start2 = dmp.diff_xIndex(diff_s_o, start2)

        if debug:
            print()
            print(start1, repr(search_text[start1 : start1 + 50]))
            print(patch.start1, repr(original_text[patch.start1 : patch.start1 + 50]))
            print(patch.diffs)
            print()

    return patches


def relative_indent(texts: list[str]) -> tuple[RelativeIndenter, list[str]]:
    """Convert texts to relative indentation format."""
    ri = RelativeIndenter(texts)
    texts = list(map(ri.make_relative, texts))
    return ri, texts


LINE_PADDING = 100


def line_pad(text: str) -> str:
    """Add newline padding before and after text."""
    padding = "\n" * LINE_PADDING
    return padding + text + padding


def line_unpad(text: str) -> str | None:
    """Remove newline padding from text, returning None if padding is invalid."""
    if set(text[:LINE_PADDING] + text[-LINE_PADDING:]) != set("\n"):
        return None
    return text[LINE_PADDING:-LINE_PADDING]


def dmp_apply(texts: list[str], remap: bool = True) -> str | None:
    """Apply diff-match-patch to transform original text using search/replace pair."""
    debug = False

    search_text, replace_text, original_text = texts

    dmp = diff_match_patch()
    dmp.Diff_Timeout = 5

    if remap:
        dmp.Match_Threshold = 0.95
        dmp.Match_Distance = 500
        dmp.Match_MaxBits = 128
        dmp.Patch_Margin = 32
    else:
        dmp.Match_Threshold = 0.5
        dmp.Match_Distance = 100_000
        dmp.Match_MaxBits = 32
        dmp.Patch_Margin = 8

    diff = dmp.diff_main(search_text, replace_text, False)
    dmp.diff_cleanupSemantic(diff)
    dmp.diff_cleanupEfficiency(diff)

    patches = dmp.patch_make(search_text, diff)

    if debug:
        html = dmp.diff_prettyHtml(diff)
        Path("tmp.search_replace_diff.html").write_text(html)
        for d in diff:
            print(d[0], repr(d[1]))
        for patch in patches:
            start1 = patch.start1
            print()
            print(start1, repr(search_text[start1 : start1 + 10]))
            print(start1, repr(replace_text[start1 : start1 + 10]))
            print(patch.diffs)

    if remap:
        patches = map_patches(texts, patches, debug)

    new_text, success = dmp.patch_apply(patches, original_text)
    all_success = False not in success

    if debug:
        print(dmp.patch_toText(patches))
        dump(success)
        dump(all_success)

    if not all_success:
        return None

    return new_text


def lines_to_chars(lines: str, mapping: list[str]) -> str:
    """Convert a list of lines back into a single string, using the provided mapping"""
    new_text_list = []
    for char in lines:
        new_text_list.append(mapping[ord(char)])
    new_text = "".join(new_text_list)
    return new_text


def dmp_lines_apply(texts: list[str]) -> str | None:
    """Apply diff-match-patch at line granularity for better matching."""
    debug = False

    for t in texts:
        assert t.endswith("\n"), t

    search_text, replace_text, original_text = texts

    dmp = diff_match_patch()
    dmp.Diff_Timeout = 5
    dmp.Match_Threshold = 0.1
    dmp.Match_Distance = 100_000
    dmp.Match_MaxBits = 32
    dmp.Patch_Margin = 1

    all_text = search_text + replace_text + original_text
    all_lines, _, mapping = dmp.diff_linesToChars(all_text, "")
    assert len(all_lines) == len(all_text.splitlines())

    search_num = len(search_text.splitlines())
    replace_num = len(replace_text.splitlines())
    original_num = len(original_text.splitlines())

    search_lines = all_lines[:search_num]
    replace_lines = all_lines[search_num : search_num + replace_num]
    original_lines = all_lines[search_num + replace_num :]

    assert len(search_lines) == search_num
    assert len(replace_lines) == replace_num
    assert len(original_lines) == original_num

    diff_lines_ = dmp.diff_main(search_lines, replace_lines, False)
    dmp.diff_cleanupSemantic(diff_lines_)
    dmp.diff_cleanupEfficiency(diff_lines_)

    patches = dmp.patch_make(search_lines, diff_lines_)

    if debug:
        diff2 = list(diff_lines_)
        dmp.diff_charsToLines(diff2, mapping)
        html = dmp.diff_prettyHtml(diff2)
        Path("tmp.search_replace_diff.html").write_text(html)
        for d in diff2:
            print(d[0], repr(d[1]))

    new_lines, success = dmp.patch_apply(patches, original_lines)
    new_text = lines_to_chars(new_lines, mapping)

    all_success = False not in success

    if debug:
        dump(success)
        dump(all_success)

    if not all_success:
        return None

    return new_text


def diff_lines(search_text: str, replace_text: str) -> list:
    """Generate a unified diff between search and replace text at line level."""
    dmp = diff_match_patch()
    dmp.Diff_Timeout = 5

    search_lines, replace_lines, mapping = dmp.diff_linesToChars(search_text, replace_text)

    diff_lines_ = dmp.diff_main(search_lines, replace_lines, False)
    dmp.diff_cleanupSemantic(diff_lines_)
    dmp.diff_cleanupEfficiency(diff_lines_)

    diff = list(diff_lines_)
    dmp.diff_charsToLines(diff, mapping)

    udiff = []
    for d, lines in diff:
        if d < 0:
            d = "-"
        elif d > 0:
            d = "+"
        else:
            d = " "
        for line in lines.splitlines(keepends=True):
            udiff.append(d + line)

    return udiff


# ---------------------------------------------------------------------------
# Core search/replace strategies
# ---------------------------------------------------------------------------


def search_and_replace(texts: list[str]) -> str | None:
    """Perform a literal search and replace."""

    search_text, replace_text, original_text = texts

    num = original_text.count(search_text)
    if num == 0:
        return None

    new_text = original_text.replace(search_text, replace_text)
    return new_text


def git_cherry_pick_osr_onto_o(texts: list[str]) -> str | None:
    """Apply search/replace via git cherry-pick: O->S->R, then cherry-pick R onto O."""
    if git is None:
        return None

    search_text, replace_text, original_text = texts

    with GitTemporaryDirectory() as dname:
        repo = git.Repo(dname)
        fname = Path(dname) / "file.txt"

        # Make O->S->R
        fname.write_text(original_text)
        repo.git.add(str(fname))
        repo.git.commit("-m", "original")
        original_hash = repo.head.commit.hexsha

        fname.write_text(search_text)
        repo.git.add(str(fname))
        repo.git.commit("-m", "search")

        fname.write_text(replace_text)
        repo.git.add(str(fname))
        repo.git.commit("-m", "replace")
        replace_hash = repo.head.commit.hexsha

        # go back to O
        repo.git.checkout(original_hash)

        # cherry pick R onto original
        try:
            repo.git.cherry_pick(replace_hash, "--minimal")
        except (ODBError, GitError):
            # merge conflicts!
            return None

        new_text = fname.read_text()
        return new_text


def git_cherry_pick_sr_onto_so(texts: list[str]) -> str | None:
    """Apply search/replace via git cherry-pick: S->R branch, cherry-pick onto S->O."""
    if git is None:
        return None

    search_text, replace_text, original_text = texts

    with GitTemporaryDirectory() as dname:
        repo = git.Repo(dname)
        fname = Path(dname) / "file.txt"

        fname.write_text(search_text)
        repo.git.add(str(fname))
        repo.git.commit("-m", "search")
        search_hash = repo.head.commit.hexsha

        # make search->replace
        fname.write_text(replace_text)
        repo.git.add(str(fname))
        repo.git.commit("-m", "replace")
        replace_hash = repo.head.commit.hexsha

        # go back to search
        repo.git.checkout(search_hash)

        # make search->original
        fname.write_text(original_text)
        repo.git.add(str(fname))
        repo.git.commit("-m", "original")

        # cherry pick replace onto original
        try:
            repo.git.cherry_pick(replace_hash, "--minimal")
        except (ODBError, GitError):
            # merge conflicts!
            return None

        new_text = fname.read_text()
        return new_text


class SearchTextNotUniqueError(ValueError):
    """Raised when search text matches multiple locations in the original."""

    pass


# ---------------------------------------------------------------------------
# Preprocessor / strategy tables
# ---------------------------------------------------------------------------

all_preprocs = [
    # (strip_blank_lines, relative_indent, reverse_lines)
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (True, True, False),
]

always_relative_indent = [
    (False, True, False),
    (True, True, False),
]

editblock_strategies = [
    (search_and_replace, all_preprocs),
    (git_cherry_pick_osr_onto_o, all_preprocs),
    (dmp_lines_apply, all_preprocs),
]

never_relative = [
    (False, False),
    (True, False),
]

udiff_strategies = [
    (search_and_replace, all_preprocs),
    (git_cherry_pick_osr_onto_o, all_preprocs),
    (dmp_lines_apply, all_preprocs),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flexible_search_and_replace(
    texts: list[str],
    strategies: list | None = None,
) -> str | None:
    """Try a series of search/replace methods, starting from the most
    literal interpretation of search_text. If needed, progress to more
    flexible methods, which can accommodate divergence between
    search_text and original_text and yet still achieve the desired
    edits.
    """
    if strategies is None:
        strategies = editblock_strategies

    for strategy, preprocs in strategies:
        for preproc in preprocs:
            res = try_strategy(texts, strategy, preproc)
            if res:
                return res

    return None


def reverse_lines(text: str) -> str:
    """Reverse the order of lines in text."""
    lines = text.splitlines(keepends=True)
    lines.reverse()
    return "".join(lines)


def try_strategy(
    texts: list[str],
    strategy: Any,
    preproc: tuple,
) -> str | None:
    """Apply a search/replace strategy with the given preprocessing options."""
    preproc_strip_blank_lines, preproc_relative_indent, preproc_reverse = preproc
    ri = None

    if preproc_strip_blank_lines:
        texts = strip_blank_lines(texts)
    if preproc_relative_indent:
        ri, texts = relative_indent(texts)
    if preproc_reverse:
        texts = list(map(reverse_lines, texts))

    res = strategy(texts)

    if res and preproc_reverse:
        res = reverse_lines(res)

    if res and preproc_relative_indent:
        try:
            res = ri.make_absolute(res)  # type: ignore[union-attr]
        except ValueError:
            return None

    return res


def strip_blank_lines(texts: list[str]) -> list[str]:
    """Strip leading and trailing blank lines from each text."""
    texts = [text.strip("\n") + "\n" for text in texts]
    return texts
