"""Shortest reference-chain queries over the document reference graph.

Reuses the analyzer's graph construction, so chain queries see exactly the
same edges as validation (same Markdown parsing and relative-link resolution
rules; fragment-only and external links never create edges).
"""
from __future__ import annotations

import heapq

from .analyzer import analyze


def shortest_reference_path(
    documents: dict[str, str], start: str, target: str
) -> list[str] | None:
    """Return the shortest Markdown reference chain from ``start`` to ``target``.

    The chain is a list of document paths, ``start`` first and ``target``
    last; a chain of ``[start]`` means ``start == target``. Among several
    equally short chains the lexicographically smallest path sequence is
    returned, so the result is unique. Returns ``None`` when ``target`` is
    not reachable from ``start``. Cycles in the graph cannot cause
    non-termination: every document is settled at most once.
    """
    graph = analyze(documents).graph

    # Dijkstra with priority (distance, path): uniform edge cost keeps the
    # distance minimal and the path tuple itself breaks ties lexicographically.
    # Extending a key strictly increases it and preserves order, so candidate
    # chains for any document are generated in non-decreasing order and the
    # first one popped is final.
    best: dict[str, tuple[int, tuple[str, ...]]] = {start: (0, (start,))}
    heap: list[tuple[int, tuple[str, ...]]] = [(0, (start,))]
    while heap:
        dist, path = heapq.heappop(heap)
        node = path[-1]
        if (dist, path) != best[node]:
            continue  # stale heap entry superseded by a better chain
        if node == target:
            return list(path)
        for nxt in sorted(graph[node]):
            candidate = (dist + 1, path + (nxt,))
            if nxt not in best or candidate < best[nxt]:
                best[nxt] = candidate
                heapq.heappush(heap, candidate)
    return None
