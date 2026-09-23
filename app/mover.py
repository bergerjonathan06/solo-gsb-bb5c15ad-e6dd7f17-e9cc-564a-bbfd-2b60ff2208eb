"""Document-move preview: rewrite links around moving one document.

Moving a document from ``src`` to ``dest`` changes two kinds of links; both
must be rewritten so every link still resolves to the same document:

1. In every *other* document, internal links that resolved to ``src`` now
   resolve to ``dest``.
2. In the moved document itself, every outbound relative link is recomputed
   relative to its new directory.

Only link destinations are rewritten: fragments, titles, angle-bracket
wrapping and the inline/image/reference-style link *forms* are preserved.
Link extraction reuses :mod:`app.parser`, so links inside fenced/indented
code blocks and inline code are skipped, as are external links and
fragment-only in-page links.
"""
from __future__ import annotations

import posixpath
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import unquote

from .analyzer import is_external, resolve_target, split_fragment
from .parser import ParsedDoc, parse_markdown


@dataclass
class _Edit:
    line: int                      # 1-based
    start: int
    end: int
    text: str


def _encode_non_angle(path_part: str) -> str:
    """Encode spaces in a destination written without angle brackets."""
    return path_part.replace(" ", "%20")


def _rewrite_destination(raw: str, new_path: str, had_angles: bool) -> str:
    """Replace the path part of ``raw`` with ``new_path``.

    The fragment is preserved verbatim and the original wrapping is kept:
    angle-bracket destinations keep spaces literal, bare destinations get
    spaces percent-encoded (CommonMark-safe).
    """
    _, frag, had_frag = split_fragment(raw)
    path_part = new_path if had_angles else _encode_non_angle(new_path)
    if had_frag:
        path_part += "#" + (frag or "")
    return path_part


def _new_path_for_link(
    *,
    owner: str,
    raw: str,
    resolved: str,
    src: str,
    dest: str,
) -> str:
    """Path the link should carry after the move, in its original form."""
    target_part, _, _ = split_fragment(raw)
    was_root_absolute = target_part.startswith("/")
    new_target = dest if resolved == src else resolved

    if was_root_absolute:
        # root-relative links are form-invariant: only the stored path changes
        return "/" + new_target

    new_owner = dest if owner == src else owner
    start_dir = posixpath.dirname(new_owner) or "."
    return posixpath.relpath(new_target, start_dir)


def build_move_preview(
    documents: dict[str, str], src: str, dest: str
) -> dict:
    """Compute the link rewrites implied by moving ``src`` to ``dest``.

    ``documents`` is not mutated and nothing is persisted. Returns one entry
    per changed document (the moved document is always present under
    ``dest``) with per-line changes and the full rewritten content.
    """
    parsed: dict[str, ParsedDoc] = {
        p: parse_markdown(p, c) for p, c in documents.items()
    }

    edits: dict[str, list[_Edit]] = defaultdict(list)
    link_count = 0

    for path, pdoc in parsed.items():
        for link in pdoc.links:
            raw = link.raw_target
            if is_external(raw):
                continue
            target_part, _, _ = split_fragment(raw)
            if target_part == "":
                continue  # fragment-only in-page link: directory independent
            resolved = resolve_target(path, unquote(target_part))

            # others: only links pointing at the moved document; the moved
            # document itself: every outbound link must be re-based
            if path != src and resolved != src:
                continue

            new_path = _new_path_for_link(
                owner=path,
                raw=raw,
                resolved=resolved,
                src=src,
                dest=dest,
            )
            line_text = pdoc.lines[link.line - 1]
            had_angles = (
                line_text[link.start - 1] == "<"
                and line_text[link.end] == ">"
            )
            new_dest = _rewrite_destination(raw, new_path, had_angles)
            if new_dest == raw:
                continue
            edits[path].append(_Edit(link.line, link.start, link.end, new_dest))
            link_count += 1

    # the moved document is always reported (its path changed), even if its
    # own outbound links needed no rewrite
    affected = set(edits) | {src}
    changed_documents = []

    for path in sorted(affected, key=lambda p: dest if p == src else p):
        pdoc = parsed[path]
        out_path = dest if path == src else path
        lines = list(pdoc.lines)

        by_line: dict[int, list[_Edit]] = defaultdict(list)
        for e in edits[path]:
            by_line[e.line].append(e)

        changes = []
        for line_no in sorted(by_line):
            original = lines[line_no - 1]
            cur = original
            # apply right-to-left so original spans stay valid
            for e in sorted(by_line[line_no], key=lambda x: x.start, reverse=True):
                cur = cur[: e.start] + e.text + cur[e.end :]
            lines[line_no - 1] = cur
            changes.append(
                {
                    "line": line_no,
                    "kind": "link-target",
                    "before": original,
                    "after": cur,
                }
            )

        changed_documents.append(
            {
                "path": out_path,
                "moved": path == src,
                "changes": changes,
                "new_content": pdoc.newline.join(lines),
            }
        )

    return {
        "source": src,
        "destination": dest,
        "documents": changed_documents,
        "stats": {
            "documents_changed": len(changed_documents),
            "links_rewritten": link_count,
        },
    }
