import re


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_author_name(name: str) -> str:
    """Normalize author names conservatively for deterministic grouping."""
    return _WHITESPACE_RE.sub(" ", name.strip())


def canonicalize_authors(authors: list[str]) -> list[str]:
    """Drop empty names and duplicates while preserving order."""
    seen: set[str] = set()
    canonical: list[str] = []
    for author in authors:
        clean = normalize_author_name(author)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        canonical.append(clean)
    return canonical
