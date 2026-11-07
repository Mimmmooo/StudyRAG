"""Conservative checks for factual author questions and cited evidence."""

import re
import unicodedata

_AUTHOR_WORDS = ("লেখক", "রচয়িতা", "author", "written by")


def _normalize(value: str) -> str:
    """Normalize Bengali Unicode variants, spacing and punctuation."""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("ড়", "ড়").replace("ঢ়", "ঢ়")
    return re.sub(r"\s+", " ", value).strip()


def is_author_question(question: str) -> bool:
    """Detect a request for a named author rather than a general explanation."""
    q = _normalize(question)
    return any(word in q for word in _AUTHOR_WORDS)


def author_supported(answer: str, evidence: str) -> bool:
    """Require the stated author to appear verbatim in the cited passage.

    This is intentionally conservative: OCR corruption can cause an abstention
    rather than presenting a confidently wrong attribution.
    """
    plain = re.sub(r"\[\d+\]", "", answer)
    plain = re.sub(r"(?i)\b(?:page|পৃষ্ঠা)\s*\d+\b", "", plain)
    candidates = re.findall(
        r"(?:লেখক|রচয়িতা)\s*(?:হলেন|হলো|হচ্ছেন|ছিলেন|হল|:|—|–|-)?\s*"
        r"([\u0980-\u09ff]+(?:\s+[\u0980-\u09ff]+){0,2})",
        plain,
    )
    candidates += re.findall(
        r"(?:author|written by)\s*(?:is|:)?\s*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
        plain,
        flags=re.IGNORECASE,
    )
    text = _normalize(evidence)
    return any(
        len(_normalize(candidate)) >= 4 and _normalize(candidate) in text
        for candidate in candidates
    )


def numeric_detail_supported(answer: str, evidence: str) -> bool:
    """Reject unsupported digits, such as invented textbook page numbers."""
    answer_numbers = set(re.findall(r"\d+", re.sub(r"\[\d+\]", "", answer)))
    return answer_numbers.issubset(set(re.findall(r"\d+", evidence)))


def explicit_author_from_passage(question: str, passage: str) -> str | None:
    """Extract only an explicitly labeled author next to the queried title.

    Scanned pages often contain OCR mistakes. Never treat a character's name
    in story prose as the author, and never use the model's world knowledge.
    """
    q = _normalize(question)
    title_match = re.search(
        r"([\u0980-\u09ff]{3,})\\s+(?:গল্প|কবিতা|প্রবন্ধ|রচনা)(?:টির|টির|ের)?",
        q,
    )
    if not title_match:
        return None
    title = title_match.group(1)
    lines = [_normalize(line) for line in passage.splitlines() if line.strip()]
    candidates: set[str] = set()
    for index, line in enumerate(lines):
        if title not in line:
            continue
        for nearby in lines[index : index + 4]:
            labeled = re.search(
                r"(?:লেখক|রচয়িতা|লিখেছেন|রচনা করেছেন|author|by)\\s*[:：-]?\\s*"
                r"([\u0980-\u09ff]+(?:\\s+[\u0980-\u09ff]+){0,2})",
                nearby,
            )
            if labeled:
                candidates.add(labeled.group(1))
        # Typical title/author layout: title on its own line, then a
        # short, standalone name on the following line.
        if line == title and index + 1 < len(lines):
            following = lines[index + 1]
            if (
                re.fullmatch(r"[\u0980-\u09ff]+(?:\\s+[\u0980-\u09ff]+){1,2}", following)
                and not any(mark in following for mark in ("।", "?", "!", ":"))
            ):
                candidates.add(following)
    return next(iter(candidates)) if len(candidates) == 1 else None
