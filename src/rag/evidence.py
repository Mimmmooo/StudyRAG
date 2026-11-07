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
