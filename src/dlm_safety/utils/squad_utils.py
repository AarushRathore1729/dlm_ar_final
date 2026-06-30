"""
Shared SQuAD extraction utilities.

Used by both ``checklist_adapter.py`` (direct path) and
``run_checklist.py`` (official path) so extraction logic is never duplicated.
"""

import re
from typing import Optional, Tuple

__all__ = [
    "SQUAD_SENTINEL",
    "SQUAD_CUE_PHRASES",
    "extract_squad_span",
    "is_valid_squad_span",
]

# Sentinel for answers that could not be extracted or repaired.
# Must be non-empty (validator rejects empty lines) and must not
# accidentally match any real gold label.
SQUAD_SENTINEL = "__INVALID_SPAN__"

# Ordered longest-first so greedy stripping works correctly.
SQUAD_CUE_PHRASES: tuple[str, ...] = (
    "based on the passage, the answer is",
    "according to the passage, the answer is",
    "the answer to this question is",
    "the answer to the question is",
    "the correct answer is",
    "the answer is",
    "answer:",
    "it's",
    "it is",
)

_MAX_SPAN_WORDS = 6

_ALLOWED_TITLE_CONNECTORS = {
    "of", "the", "and", "de", "da", "del", "van", "von",
}

_WEAK_SINGLE_TOKEN_SPANS = {
    "a", "an", "the", "about", "above", "after", "all", "also", "and",
    "any", "are", "around", "as", "at", "be", "because", "before", "below",
    "between", "both", "but", "by", "during", "each", "for", "from", "how",
    "if", "in", "into", "is", "it", "its", "more", "most", "no", "not", "of",
    "on", "one", "or", "other", "out", "over", "so", "than", "that", "the",
    "their", "there", "these", "they", "this", "those", "through", "to",
    "under", "until", "up", "very", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "would", "yes",
}

_ABSTENTION_PHRASES = (
    "there is no information",
    "there is not enough information",
    "no information",
    "not enough information",
    "cannot determine",
    "can't determine",
    "can not determine",
    "cannot be determined",
    "can't be determined",
    "can not be determined",
    "not stated",
    "is not stated",
    "not mentioned",
    "is not mentioned",
    "unclear",
    "unknown",
)

_WRAPPER_CHARS = "\"'`[](){}"

_COLOR_WORDS = {
    "black", "blue", "brown", "gold", "gray", "green", "orange", "pink",
    "purple", "red", "silver", "white", "yellow",
}

_SHAPE_WORDS = {
    "circular", "oval", "rectangular", "round", "square", "triangular",
}

_SIZE_WORDS = {
    "big", "enormous", "gigantic", "huge", "large", "little", "massive",
    "miniature", "small", "tiny",
}

_AGE_WORDS = {
    "ancient", "elderly", "middle-aged", "new", "old", "older", "young",
    "younger",
}


def _clean_candidate(text: str) -> str:
    """Normalize a candidate span without changing its core wording."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = cleaned.strip(_WRAPPER_CHARS).strip()
    cleaned = re.sub(r"^[,;:.\-]+", "", cleaned).strip()
    cleaned = re.sub(r"[,;:.!?]+$", "", cleaned).strip()
    return cleaned


def _contains_abstention(text: str) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in _ABSTENTION_PHRASES)


def _span_pattern(span: str) -> str:
    escaped = re.escape(span)
    left = r"(?<!\w)" if span and span[0].isalnum() else ""
    right = r"(?!\w)" if span and span[-1].isalnum() else ""
    return left + escaped + right


def _find_passage_match(span: str, passage: str) -> Optional[re.Match[str]]:
    if not span or not passage:
        return None
    return re.search(_span_pattern(span), passage, flags=re.IGNORECASE)


def _looks_like_salient_substring(candidate: str) -> bool:
    """Require stronger evidence before mining a span from explanatory text."""
    words = candidate.split()
    if not words:
        return False

    if len(words) == 1:
        token = words[0]
        return token[:1].isupper() or any(ch.isdigit() for ch in token)

    salient = 0
    for word in words:
        stripped = word.strip(".,;:!?")
        if not stripped:
            return False
        if stripped[:1].isupper() or any(ch.isdigit() for ch in stripped):
            salient += 1
            continue
        if stripped.lower() in _ALLOWED_TITLE_CONNECTORS:
            continue
        return False
    return salient > 0


def _extract_candidate_from_cue(after_cue: str) -> str:
    """Prefer a direct answer line after a cue, then a short phrase."""
    lines = [line.strip() for line in after_cue.split("\n") if line.strip()]
    for line in lines:
        candidate = _clean_candidate(line)
        if candidate:
            return candidate
    candidate = _clean_candidate(re.split(r"[.;:!?]", after_cue, maxsplit=1)[0])
    return candidate


def _strip_article(candidate: str) -> str:
    words = candidate.split()
    if len(words) >= 2 and words[0].lower() in {"a", "an", "the"}:
        remainder = words[1]
        if remainder and remainder[0].isupper():
            return candidate
        return " ".join(words[1:])
    return candidate


def _question_asks_who(question: str) -> bool:
    q = (question or "").strip().lower()
    return q.startswith("who ")


def _match_comparative_statement(text: str) -> Optional[tuple[str, str]]:
    match = re.match(
        r"^([A-Z][\w'`-]*(?:\s+[A-Z][\w'`-]*)*)\s+is\s+.+?\s+than\s+([A-Z][\w'`-]*(?:\s+[A-Z][\w'`-]*)*)$",
        _clean_candidate(text),
    )
    if match is None:
        return None
    return match.group(1), match.group(2)


def _attribute_reduction(candidate: str, question: str) -> Optional[str]:
    q = (question or "").lower()
    tokens = [_clean_candidate(tok) for tok in candidate.split()]
    tokens = [tok for tok in tokens if tok]
    if len(tokens) < 2:
        return None

    lowered = [tok.lower() for tok in tokens]

    if "what color" in q:
        for token, lowered_token in zip(tokens, lowered):
            if lowered_token in _COLOR_WORDS:
                return token
        return None

    if "what shape" in q:
        for token, lowered_token in zip(tokens, lowered):
            if lowered_token in _SHAPE_WORDS:
                return token
        return None

    if "what size" in q:
        for token, lowered_token in zip(tokens, lowered):
            if lowered_token in _SIZE_WORDS:
                return token
        return None

    if "what age" in q or "how old" in q:
        for token, lowered_token in zip(tokens, lowered):
            if lowered_token in _AGE_WORDS:
                return token
        return None

    if "profession" in q or "occupation" in q or "job" in q:
        for token in reversed(tokens):
            if token.lower() not in {"a", "an", "the"}:
                return token
        return None

    if "nationality" in q or "country" in q:
        for token in tokens:
            if token.lower() not in {"a", "an", "the"}:
                return token
        return None

    return None


def _who_question_reduction(candidate: str, passage: str, question: str) -> Optional[str]:
    if not _question_asks_who(question):
        return None

    comparative = _match_comparative_statement(candidate)
    if comparative is not None:
        q = (question or "").lower()
        left, right = comparative
        if " less " in f" {q} " and " more " not in f" {q} ":
            return right
        if " more " in f" {q} " and " less " not in f" {q} ":
            return left

    subject_match = re.match(
        r"^([A-Z][\w'`-]*(?:\s+[A-Z][\w'`-]*)*)\s+is\b",
        candidate,
    )
    if subject_match is not None:
        return subject_match.group(1)

    return None


def _question_aware_reduce(candidate: str, passage: str, question: str) -> tuple[str, Optional[str]]:
    """Reduce over-broad but valid spans using only question/passage context."""
    candidate = _clean_candidate(candidate)
    if not candidate:
        return candidate, None

    stripped = _strip_article(candidate)
    if stripped != candidate and is_valid_squad_span(stripped, passage):
        return stripped, "article_strip"

    attribute = _attribute_reduction(candidate, question)
    if attribute and is_valid_squad_span(attribute, passage):
        return attribute, "attribute"

    who_reduced = _who_question_reduction(candidate, passage, question)
    if who_reduced and is_valid_squad_span(who_reduced, passage):
        if _match_comparative_statement(candidate):
            return who_reduced, "comparative_who"
        return who_reduced, "who_subject"

    return candidate, None


def _finalize_candidate(
    candidate: str,
    passage: str,
    question: str,
    trace: dict,
    extraction_mode: str,
    max_answer_chars: int,
    max_span_words: int,
) -> Optional[str]:
    candidate = _clean_candidate(candidate)
    if not candidate or len(candidate) > max_answer_chars:
        return None

    orig_valid = is_valid_squad_span(
        candidate,
        passage,
        max_answer_chars=max_answer_chars,
        max_span_words=max_span_words,
    )

    reduced, rule = _question_aware_reduce(candidate, passage, question)
    if rule is not None and reduced != candidate:
        trace["extraction_mode"] = "question_aware_reduction"
        trace["base_extraction_mode"] = extraction_mode
        trace["question_rule"] = rule
        return _passage_cased(reduced.lower(), passage) if passage else reduced

    if not orig_valid:
        return None

    trace["extraction_mode"] = extraction_mode
    return _passage_cased(candidate.lower(), passage) if passage else candidate


def is_valid_squad_span(
    answer: str,
    passage: str,
    max_answer_chars: int = 128,
    max_span_words: int = _MAX_SPAN_WORDS,
) -> bool:
    """Check whether *answer* is a plausible short extractive span."""
    ans = _clean_candidate(answer)
    if not ans:
        return False
    if ans == SQUAD_SENTINEL:
        return False
    if len(ans) < 2:
        return False
    if len(ans) > max_answer_chars:
        return False
    words = ans.split()
    if len(words) > max_span_words:
        return False
    if len(words) == 1 and words[0].lower() in _WEAK_SINGLE_TOKEN_SPANS:
        return False
    passage = (passage or "").strip()
    if not passage:
        # No passage to validate against — accept if length-valid.
        return True
    return _find_passage_match(ans, passage) is not None


def _passage_cased(span_lower: str, passage: str) -> str:
    """Return the passage-cased version of *span_lower*.

    If *span_lower* (already lowered) appears in *passage* (case-insensitive),
    return the corresponding slice of *passage* so the casing matches gold.
    """
    match = _find_passage_match(span_lower, passage)
    if match is None:
        return span_lower  # fallback: return as-is
    return passage[match.start() : match.end()]


def extract_squad_span(
    raw_answer: str,
    passage: str,
    question: str = "",
    max_answer_chars: int = 128,
    max_span_words: int = _MAX_SPAN_WORDS,
) -> Tuple[str, dict]:
    """Deterministic span extraction from a potentially verbose model answer.

    Returns ``(span, trace)`` where *trace* is a dict with extraction metadata:
    - ``extraction_mode``: one of ``direct``, ``first_line``, ``cue_stripped``,
      ``passage_substring``, ``failed``
    - ``stripped_cue``: the cue phrase that was stripped, if any
    """
    passage = (passage or "").strip()
    trace: dict = {
        "extraction_mode": "failed",
        "stripped_cue": None,
        "base_extraction_mode": None,
        "question_rule": None,
    }

    # --- Step 0: basic cleanup ---
    raw_lines = [
        re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        for line in (raw_answer or "").replace("\r", "\n").split("\n")
    ]
    lines = [line for line in raw_lines if line]
    cleaned = "\n".join(lines)
    cleaned_single_line = _clean_candidate(" ".join(lines))

    if not cleaned_single_line:
        return SQUAD_SENTINEL, trace

    # --- Step 1: direct valid span ---
    span = _finalize_candidate(
        cleaned_single_line,
        passage,
        question,
        trace,
        "direct",
        max_answer_chars,
        max_span_words,
    )
    if span is not None:
        return span, trace

    # --- Step 2: final answer line ---
    for line in reversed(lines):
        candidate = _clean_candidate(line)
        span = _finalize_candidate(
            candidate,
            passage,
            question,
            trace,
            "final_line",
            max_answer_chars,
            max_span_words,
        )
        if span is not None:
            return span, trace

    # --- Step 3: first line ---
    first_line = _clean_candidate(lines[0]) if lines else ""
    span = _finalize_candidate(
        first_line,
        passage,
        question,
        trace,
        "first_line",
        max_answer_chars,
        max_span_words,
    )
    if span is not None:
        return span, trace

    # --- Step 4: cue-phrase stripping ---
    text_lower = cleaned.lower()
    for cue in SQUAD_CUE_PHRASES:
        idx = text_lower.find(cue)
        if idx == -1:
            continue
        after_cue = cleaned[idx + len(cue) :].strip()
        candidate = _extract_candidate_from_cue(after_cue)
        trace["stripped_cue"] = cue
        span = _finalize_candidate(
            candidate,
            passage,
            question,
            trace,
            "cue_stripped",
            max_answer_chars,
            max_span_words,
        )
        if span is not None:
            trace["stripped_cue"] = cue
            return span, trace
        trace["stripped_cue"] = None

    # --- Step 5: abstention guard before fallback mining ---
    if _contains_abstention(cleaned_single_line):
        return SQUAD_SENTINEL, trace

    # --- Step 6: conservative passage substring fallback ---
    if passage:
        best_span = _find_conservative_passage_span(
            cleaned_single_line, passage, max_span_words=min(4, max_span_words)
        )
        if best_span:
            trace["extraction_mode"] = "passage_substring"
            return best_span, trace

    return SQUAD_SENTINEL, trace


def _find_conservative_passage_span(
    text: str,
    passage: str,
    max_span_words: int = 4,
) -> Optional[str]:
    """Find a strongly-supported span from explanatory text.

    This fallback is intentionally conservative. It only mines spans that look
    like salient answer candidates, such as names or dates, and prefers later
    occurrences in the response so final answer mentions beat quoted evidence.
    """
    words = text.split()
    if not words:
        return None

    best: Optional[str] = None
    best_score: tuple[int, int] | None = None

    for span_len in range(min(max_span_words, len(words)), 0, -1):
        for start in range(len(words) - span_len + 1):
            candidate = _clean_candidate(" ".join(words[start : start + span_len]))
            if not candidate:
                continue
            if not _looks_like_salient_substring(candidate):
                continue
            if not is_valid_squad_span(candidate, passage, max_span_words=max_span_words):
                continue
            score = (span_len, start)
            if best_score is None or score > best_score:
                best = _passage_cased(candidate.lower(), passage)
                best_score = score

    if best is not None:
        return best

    return None
