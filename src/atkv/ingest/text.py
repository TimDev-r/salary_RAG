"""Text packing shared by every ingest path.

Shared deliberately. When only the KV path packed its prose, 24% of the law
chunks silently exceeded the embedding model's 512-token window and were
truncated -- one of them losing 2,608 tokens. No error is raised for that: the
model embeds the first 512 tokens and returns a perfectly ordinary vector for
a fragment of the text you believed you had indexed.

Any new source must route through here rather than grow its own packer.
"""

from __future__ import annotations

import re

# e5 truncates at 512 tokens. German runs roughly 4 chars/token, so ~2000 chars
# is the ceiling; 1400 leaves headroom for the heading prefix we prepend, and
# for tokenizer variance on compound-heavy legal text.
MAX_PROSE_CHARS = 1250

# Block boundaries: "(1)", "(2)" and roman subsection markers "I.", "II.".
# Splitting here rather than at a character count keeps a provision whole, so a
# chunk is never half a legal rule.
BLOCK_START = re.compile(r"^\s*(?:\(\d+[a-z]?\)|[IVX]+\.)\s")


def blocks(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if BLOCK_START.match(line) and cur:
            out.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return [b for b in (b.strip() for b in out) if b]


def pack(parts: list[str], limit: int = MAX_PROSE_CHARS) -> list[str]:
    """Greedily group whole blocks up to `limit`; split oversized ones on sentences."""
    out: list[str] = []
    cur = ""
    for b in parts:
        while len(b) > limit:
            cut = b.rfind(". ", 0, limit)
            cut = cut + 1 if cut > limit // 2 else limit
            head, b = b[:cut].strip(), b[cut:].strip()
            if cur:
                out.append(cur); cur = ""
            out.append(head)
        if not cur:
            cur = b
        elif len(cur) + 1 + len(b) <= limit:
            cur += "\n" + b
        else:
            out.append(cur); cur = b
    if cur:
        out.append(cur)
    return out


def split_text(text: str, limit: int = MAX_PROSE_CHARS) -> list[str]:
    return pack(blocks(text), limit)
