"""
PFD (Regulation 28) response segmentation and scoped alignment helpers.

This module is intentionally heuristic-based and profile-scoped: it is meant to
support PFD-style documents without leaking assumptions into the `explicit_recs`
pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from .types import PreprocessedText, Recommendation, Response, Span


@dataclass(frozen=True)
class PfdResponseBlock:
    header: str
    span: Span
    text: str


@dataclass(frozen=True)
class PfdScopedAlignment:
    directive: Recommendation
    status: str  # e.g. "matched_in_scope", "out_of_scope", "unknown_scope", "no_match", "deferral"
    addressees: Tuple[str, ...]
    responder_aliases: Tuple[str, ...]
    response_block: Optional[PfdResponseBlock]
    response_snippet: Optional[str]


_BULLET_CHARS = ("•", "", "·", "▪", "–", "-")


def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _lower_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def infer_responder_aliases(response_text: str) -> Set[str]:
    """
    Infer responder identity aliases (surface forms) from a response document.

    This deliberately avoids a global responder taxonomy. It returns a set of
    strings like {"Thames Valley Police", "TVP", "CTPSE", "Chief Constable"}.
    """
    aliases: Set[str] = set()
    text = response_text or ""
    if not text.strip():
        return aliases

    # Prefer the first chunk of the document (letterhead/intro) and the signature block.
    head = text[:6000]
    signature_window = ""
    sig_match = re.search(r"\bYours\s+(?:sincerely|faithfully)\b", text, flags=re.IGNORECASE)
    if sig_match:
        signature_window = text[sig_match.start() : sig_match.start() + 1200]

    combined = head + "\n\n" + signature_window
    lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]

    # Add common role indicators.
    for ln in lines:
        if re.search(r"\bChief\s+Constable\b", ln, flags=re.IGNORECASE):
            aliases.add("Chief Constable")
        if re.search(r"\bSecretary\s+of\s+State\b", ln, flags=re.IGNORECASE):
            aliases.add("Secretary of State")

    # Extract plausible organisation names from prominent lines.
    org_pattern = re.compile(
        r"\b([A-Z][A-Za-z&’'()-]+(?:\s+[A-Z][A-Za-z&’'()-]+){0,6}\s+"
        r"(?:Police|Constabulary|Department|Office|Service|Trust|Council|Ministry))\b"
    )
    for ln in lines:
        for m in org_pattern.finditer(ln):
            aliases.add(_norm_space(m.group(1)))

    # Extract acronyms (2-6 chars) that appear in prominent positions.
    for m in re.finditer(r"\b[A-Z]{2,6}\b", combined):
        aliases.add(m.group(0))

    # Some documents refer to the responder as "we"/"our force" without naming
    # the org; keep this low-signal and do not treat it as an alias.
    aliases.discard("I")
    aliases.discard("AM")
    aliases.discard("PM")
    return {a for a in aliases if len(a) >= 3}


def extract_pfd_directive_addressees(directive_text: str) -> Set[str]:
    """
    Extract addressee surface forms from a PFD directive sentence.

    Returns surface strings; later scope gating is done by overlap with
    `infer_responder_aliases(...)`.
    """
    text = directive_text or ""
    addressees: Set[str] = set()
    if not text.strip():
        return addressees

    # Pattern 1: "<Entity> should/must/is to/to <verb>"
    subj = re.search(
        r"\b((?:The\s+)?(?:Secretary\s+of\s+State\s+for\s+the\s+Home\s+Department|"
        r"Secretary\s+of\s+State\s+for\s+Justice|Chief\s+Constable(?:\s+of\s+[A-Z][A-Za-z\s]+)?|"
        r"NHS\s+England|The\s+Trust|[A-Z][A-Za-z\s]+Trust|[A-Z]{2,6}))\s+"
        r"(should|must|is\s+to|to)\b",
        text,
    )
    if subj:
        addressees.add(_norm_space(subj.group(1)))

    # Pattern 2: "I request/recommend/suggest ... by/to <Entity>"
    by_to = re.search(
        r"\b(?:recommend|request|suggest|encourage)\b[\s\S]{0,200}?\b(?:by|to)\s+"
        r"((?:[A-Z]{2,6}|[A-Z][A-Za-z&’'()-]+(?:\s+[A-Z][A-Za-z&’'()-]+){0,6}))\b",
        text,
        flags=re.IGNORECASE,
    )
    if by_to:
        addressees.add(_norm_space(by_to.group(1)))

    # Pattern 3: compound coordination "... encourage <A> and <B> to ..."
    coord = re.search(
        r"\bencourage\s+(.+?)\s+and\s+(.+?)\s+to\b",
        text,
        flags=re.IGNORECASE,
    )
    if coord:
        a = _norm_space(coord.group(1))
        b = _norm_space(coord.group(2))
        # Keep them bounded (avoid grabbing an entire paragraph).
        if 3 <= len(a) <= 80:
            addressees.add(a)
        if 3 <= len(b) <= 80:
            addressees.add(b)

    # Normalize obvious quoted role names.
    cleaned = set()
    for a in addressees:
        cleaned.add(a.strip().strip(",.;:"))
    return {a for a in cleaned if a}


def _line_index(text: str) -> List[Tuple[int, str]]:
    """Return [(start_offset, line_text), ...] for non-empty lines."""
    out: List[Tuple[int, str]] = []
    offset = 0
    for ln in text.splitlines(True):
        stripped = ln.strip("\n")
        if stripped.strip():
            out.append((offset, stripped))
        offset += len(ln)
    return out


def _extract_expected_headers_from_intro(lines: Sequence[str]) -> Set[str]:
    expected: Set[str] = set()
    for ln in lines[:40]:
        if any(b in ln for b in _BULLET_CHARS):
            tmp = ln
            for b in _BULLET_CHARS:
                tmp = tmp.replace(b, "|")
            parts = [_norm_space(p) for p in tmp.split("|")]
            for p in parts:
                # Keep short title-ish entries.
                if 2 <= len(p.split()) <= 12 and any(c.isalpha() for c in p):
                    expected.add(p)
    return expected


def segment_pfd_response_blocks(
    preprocessed: PreprocessedText,
    source_document: str,
) -> List[PfdResponseBlock]:
    """
    Segment a PFD response document into thematic blocks delimited by headings.

    This is heuristic and expects "letter-style" responses with theme headings.
    """
    text = preprocessed.text or ""
    if not text.strip():
        return []

    indexed = _line_index(text)
    lines = [ln for _, ln in indexed]
    expected_headers = _extract_expected_headers_from_intro(lines)
    strong_keywords = {
        "MAPPA",
        "Prevent",
        "Pathfinder",
        "Intelligence",
        "Operation",
        "Plato",
        "FIM",
        "CTPSE",
    }

    def is_sentence_start(fragment: str) -> bool:
        return bool(re.match(r"^(I|The|This|We|In|As)\b", fragment.strip()))

    boundaries: List[Tuple[int, str]] = []

    for (start, ln) in indexed:
        s = ln.strip()

        # Hard break if a line begins with a strong keyword (common PFD response headings).
        for kw in strong_keywords:
            if s.startswith(kw + " ") or s == kw:
                boundaries.append((start, kw if kw != "Operation" else s.split()[0] + " " + s.split()[1] if len(s.split()) > 1 else "Operation"))
                break
        else:
            # No strong-keyword hard break.
            pass

        # Split merged "Header I acknowledge..." forms using expected headers.
        for hdr in sorted(expected_headers, key=len, reverse=True):
            if s.startswith(hdr + " ") and is_sentence_start(s[len(hdr) :]):
                boundaries.append((start, hdr))
                break

        # Title-case + sentence-case collision: "Prevent I acknowledge..."
        m = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\s+(I|The|This|We)\b", s)
        if m:
            header = m.group(1)
            if header in expected_headers or header in strong_keywords:
                boundaries.append((start, header))

    # De-duplicate and sort boundaries.
    uniq: dict[int, str] = {}
    for pos, hdr in boundaries:
        uniq.setdefault(pos, hdr)
    boundary_list = sorted(uniq.items(), key=lambda x: x[0])

    if not boundary_list:
        # Fallback: one block for entire response.
        return [
            PfdResponseBlock(
                header="Response",
                span=(0, len(text)),
                text=text.strip(),
            )
        ]

    blocks: List[PfdResponseBlock] = []
    for idx, (pos, hdr) in enumerate(boundary_list):
        end = boundary_list[idx + 1][0] if idx + 1 < len(boundary_list) else len(text)
        block_text = text[pos:end].strip()
        if not block_text:
            continue
        blocks.append(PfdResponseBlock(header=_norm_space(hdr), span=(pos, end), text=block_text))
    return blocks


def _is_deferral(text: str) -> bool:
    lower = text.lower()
    triggers = [
        "responsibility of",
        "falls under the responsibility",
        "falls to",
        "i anticipate",
        "will respond",
        "not our responsibility",
    ]
    return any(t in lower for t in triggers)


def _pick_action_snippet(block_text: str) -> Optional[str]:
    """
    Pick a short snippet from a response block that looks like "action taken/proposed".
    """
    candidates = re.split(r"(?<=[.!?])\s+", _norm_space(block_text))
    markers = [
        "we have",
        "we will",
        "we are",
        "review",
        "introduced",
        "changes",
        "undertook",
        "stress-test",
        "stress test",
        "exercise",
        "commissioned",
        "in hand",
        "am confident",
        "i agree",
        "i acknowledge",
    ]
    for sent in candidates:
        s = sent.strip()
        if len(s) < 40:
            continue
        lower = s.lower()
        if any(m in lower for m in markers):
            return s
    return candidates[0].strip() if candidates else None


def align_pfd_directives_to_response_blocks(
    directives: Iterable[Recommendation],
    response_blocks: Sequence[PfdResponseBlock],
    responder_aliases: Set[str],
) -> List[PfdScopedAlignment]:
    """
    Align PFD directives to response blocks with scope gating by responder identity.
    """
    responder_norm = {a.lower() for a in responder_aliases}
    results: List[PfdScopedAlignment] = []

    for directive in directives:
        addressees = extract_pfd_directive_addressees(directive.text)
        addrs_norm = {a.lower() for a in addressees}

        in_scope = False
        if addrs_norm:
            for a in addrs_norm:
                if any(a in r or r in a for r in responder_norm):
                    in_scope = True
                    break
        else:
            # If no explicit addressee is found, keep it ambiguous.
            in_scope = True

        if not in_scope and addrs_norm:
            results.append(
                PfdScopedAlignment(
                    directive=directive,
                    status="out_of_scope",
                    addressees=tuple(sorted(addressees)),
                    responder_aliases=tuple(sorted(responder_aliases)),
                    response_block=None,
                    response_snippet=None,
                )
            )
            continue

        # Topic match by overlap with block headers and block text.
        directive_tokens = set(_lower_tokens(directive.text))
        best: Optional[PfdResponseBlock] = None
        best_score = 0.0
        for block in response_blocks:
            block_tokens = set(_lower_tokens(block.header + " " + block.text[:2000]))
            if not block_tokens:
                continue
            intersection = len(directive_tokens & block_tokens)
            union = len(directive_tokens | block_tokens) or 1
            score = intersection / union
            # Bonus if header appears directly in directive.
            if block.header and block.header.lower() in directive.text.lower():
                score += 0.15
            if score > best_score:
                best_score = score
                best = block

        if best is None or best_score < 0.05:
            results.append(
                PfdScopedAlignment(
                    directive=directive,
                    status="no_match",
                    addressees=tuple(sorted(addressees)),
                    responder_aliases=tuple(sorted(responder_aliases)),
                    response_block=None,
                    response_snippet=None,
                )
            )
            continue

        snippet = _pick_action_snippet(best.text)
        status = "matched_in_scope"
        if snippet and _is_deferral(snippet):
            status = "deferral"

        results.append(
            PfdScopedAlignment(
                directive=directive,
                status=status,
                addressees=tuple(sorted(addressees)),
                responder_aliases=tuple(sorted(responder_aliases)),
                response_block=best,
                response_snippet=snippet,
            )
        )

    return results

