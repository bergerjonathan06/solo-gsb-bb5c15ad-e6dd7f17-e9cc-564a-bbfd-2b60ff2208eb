"""Move preview: rewrite links as if one document were moved to a new path.

Two kinds of destinations are rewritten, and nothing else:

1. **Incoming links** — in every document (including the moved one),
   internal links whose destination resolves to the moved document are
   repointed at its new path: the references follow the document.
2. **Outgoing relative links of the moved document** — because its
   directory changes, every *relative* link it contains is recomputed
   against the new location, resolving to the same document as before.

Like the fix preview, only link destinations are touched. Fragments,
link titles, angle-bracket destinations and the inline/image/
reference-style link forms are preserved; fenced/indented code blocks,
inline code spans and external links are skipped (the parser never
records links there). Nothing is persisted.
"""
from __future__ import annotations

import posixpath
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import unquote

from .analyzer import is_external, resolve_target, split_fragment
from .parser import Link, ParsedDoc, parse_markdown


@dataclass
class SpanEdit:
    line: int          # 1-based
    start: int
    end: int
    text: str
    kind: str          # link-follow | link-relative


def _resolve_from_dir(holder_dir: str, target_path: str) -> str:
    holder = posixpath.join(holder_dir, "holder.md") if holder_dir else "holder.md"
    return resolve_target(holder, target_path)


def _follow_destination(raw: str, holder_dir: str, source: str, target: str) -> str | None:
    """Rewrite a link pointing at the moved document at its new location.

    Returns the new destination text (fragment preserved), or None when the
    link does not resolve to the source document.
    """
    path_part, frag, had_frag = split_fragment(raw)
    if path_part == "" or is_external(path_part):
        return None  # pure-fragment in-page link or external
    if path_part.startswith("/"):
        if _resolve_from_dir(holder_dir, path_part) != source:
            return None
        new_path = "/" + target
    else:
        if _resolve_from_dir(holder_dir, unquote(path_part)) != source:
            return None
        new_path = posixpath.relpath(target, holder_dir) if holder_dir else target
    return new_path + ("#" + frag if had_frag else "")


def _recompute_relative(raw: str, source: str, target: str, new_dir: str) -> tuple[str, str] | None:
    """Re-express an outbound link of the moved document from its new
    directory so it keeps pointing at the same document. A link that used to
    point at the document itself follows it to ``target``.

    Returns (new_path, kind) (fragment still attached) or None for external,
    pure-fragment and root-absolute links that point at another document.
    """
    path_part, frag, had_frag = split_fragment(raw)
    if path_part == "" or is_external(path_part):
        return None
    if path_part.startswith("/"):
        resolved = resolve_target(source, path_part)
        if resolved != source:
            return None  # root-absolute links are location-independent
        new_path = "/" + target
        kind = "link-follow"
    else:
        old_resolved = resolve_target(source, unquote(path_part))
        follows = old_resolved == source
        dest = target if follows else old_resolved
        new_path = posixpath.relpath(dest, new_dir) if new_dir else dest
        kind = "link-follow" if follows else "link-relative"
    return new_path + ("#" + frag if had_frag else ""), kind


def _iter_rewritable_links(pdoc: ParsedDoc):
    """Inline/image links plus every reference-style definition (used or not)."""
    yield from (l for l in pdoc.links if l.kind in ("inline", "image"))
    yield from pdoc.ref_definitions


def _wrapped_in_angles(line: str, link: Link) -> bool:
    """Whether the destination of ``link`` is wrapped in angle brackets."""
    return (
        link.start > 0
        and line[link.start - 1] == "<"
        and link.end < len(line)
        and line[link.end] == ">"
    )


def build_move_preview(
    documents: dict[str, str], source: str, target: str
) -> dict:
    """Compute the line-level rewrites of moving ``source`` to ``target``.

    ``source`` must exist and ``target`` must be free; callers validate that.
    Paths are already normalized by the import rules.
    """
    parsed = {p: parse_markdown(p, c) for p, c in documents.items()}
    new_dir = posixpath.dirname(target)

    # holder path -> edits. The moved document is keyed by its *new* path.
    edits: dict[str, list[SpanEdit]] = defaultdict(list)

    for path, pdoc in parsed.items():
        moved = path == source
        out_path = target if moved else path
        holder_dir = new_dir if moved else posixpath.dirname(path)
        lines = pdoc.lines

        for link in _iter_rewritable_links(pdoc):
            line = lines[link.line - 1]
            angles = _wrapped_in_angles(line, link)
            if moved:
                recomputed = _recompute_relative(
                    link.raw_target, source, target, new_dir
                )
                if recomputed is None:
                    continue
                new_raw, kind = recomputed
            else:
                new_raw = _follow_destination(
                    link.raw_target, holder_dir, source, target
                )
                if new_raw is None:
                    continue
                kind = "link-follow"

            # literal spaces are legal only inside angle brackets; elsewhere
            # they must be percent-encoded (unicode stays readable)
            if not angles and " " in new_raw:
                path_part, sep, frag = new_raw.partition("#")
                new_raw = path_part.replace(" ", "%20") + (sep + frag if sep else "")
            if angles:
                start, end, text = link.start - 1, link.end + 1, "<" + new_raw + ">"
            else:
                start, end, text = link.start, link.end, new_raw
            if line[start:end] == text:
                continue  # destination text unchanged (e.g. same-directory move)
            edits[out_path].append(SpanEdit(link.line, start, end, text, kind))

    changed_documents = []
    edit_count = 0
    # the moved document is always part of the preview (keyed at its new path),
    # even when no link destination changes
    affected = set(edits) | {target}
    for out_path in sorted(affected):
        src_path = source if out_path == target else out_path
        pdoc = parsed[src_path]
        lines = list(pdoc.lines)
        by_line: dict[int, list[SpanEdit]] = defaultdict(list)
        for e in edits[out_path]:
            by_line[e.line].append(e)

        changes = []
        for line_no in sorted(by_line):
            original = lines[line_no - 1]
            line_edits = by_line[line_no]
            # apply right-to-left so original spans stay valid
            cur = original
            kinds = set()
            for e in sorted(line_edits, key=lambda x: (x.start, x.end), reverse=True):
                cur = cur[: e.start] + e.text + cur[e.end :]
                kinds.add(e.kind)
            lines[line_no - 1] = cur
            changes.append(
                {
                    "line": line_no,
                    "kind": "+".join(sorted(kinds)),
                    "before": original,
                    "after": cur,
                }
            )
        edit_count += sum(len(v) for v in by_line.values())
        changed_documents.append(
            {
                "path": out_path,
                "changes": changes,
                "new_content": pdoc.newline.join(lines),
            }
        )

    return {
        "source": source,
        "target": target,
        "documents": changed_documents,
        "stats": {
            "documents_changed": len(changed_documents),
            "edits": edit_count,
        },
    }
