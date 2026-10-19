"""Parser and evaluator for the search query language."""

from __future__ import annotations

import re
import unicodedata
from itertools import combinations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse


ALLOWED_FIELDS = {"title", "author", "site", "type", "source", "year", "citations", "language", "open_access"}
ALLOWED_RECORD_TYPES = {"paper", "person", "organization", "website", "news", "social", "other"}
ALLOWED_SOURCES = {
    "arxiv", "semanticscholar", "openalex", "crossref", "pubmed", "dblp", "duckduckgo", "google",
    "academic", "all", "brave", "bravesearch",
}
_EMAIL_TERM_RE = re.compile(r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
ARABIC_FOLDS = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ؤ": "و", "ئ": "ي", "ى": "ي",
    "ک": "ك", "ی": "ي",
})


class QuerySyntaxError(ValueError):
    """Raised when a search expression is malformed or uses an unknown field."""


@dataclass(frozen=True)
class Node:
    kind: str
    value: str = ""
    field: Optional[str] = None
    children: Tuple["Node", ...] = ()


TOKEN_RE = re.compile(r'(?P<field>[A-Za-z_]+):(?P<field_value>"(?:\\.|[^"\\])*"|[^\s()]+)|(?P<phrase>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\\\'])*\')|(?P<paren>[()])|(?P<word>[^\s()]+)')


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return re.sub(r"\\([\\\"'])", r"\1", value[1:-1])
    return value


def tokenize(query: str) -> List[Tuple[str, str, Optional[str]]]:
    tokens = []
    pos = 0
    while pos < len(query):
        if query[pos].isspace():
            pos += 1
            continue
        if query[pos] in {'"', "'"}:
            quote = query[pos]
            end = pos + 1
            escaped = False
            while end < len(query):
                if query[end] == quote and not escaped:
                    break
                escaped = query[end] == "\\" and not escaped
                if query[end] != "\\":
                    escaped = False
                end += 1
            if end >= len(query):
                raise QuerySyntaxError(f"Unclosed quote at position {pos + 1}")
        field_prefix = re.match(r"[A-Za-z_]+:", query[pos:])
        if field_prefix:
            after_field = pos + field_prefix.end()
            if after_field >= len(query) or query[after_field].isspace() or query[after_field] in "()":
                raise QuerySyntaxError(f"Field '{field_prefix.group()[:-1]}' needs a value")
            if query[after_field] in {'"', "'"}:
                quote = query[after_field]
                end = after_field + 1
                escaped = False
                while end < len(query):
                    if query[end] == quote and not escaped:
                        break
                    escaped = query[end] == "\\" and not escaped
                    if query[end] != "\\":
                        escaped = False
                    end += 1
                if end >= len(query):
                    raise QuerySyntaxError(f"Unclosed quote in field at position {after_field + 1}")
        match = TOKEN_RE.match(query, pos)
        if not match:
            raise QuerySyntaxError(f"Unexpected character at position {pos + 1}")
        if match.group("field") is not None:
            field = match.group("field").casefold()
            if field not in ALLOWED_FIELDS:
                raise QuerySyntaxError(f"Unknown field '{field}' at position {pos + 1}")
            value = _unquote(match.group("field_value"))
            tokens.append(("TERM", value, field))
        elif match.group("phrase") is not None:
            tokens.append(("TERM", _unquote(match.group("phrase")), None))
        elif match.group("paren") is not None:
            value = match.group("paren")
            tokens.append(("LPAREN" if value == "(" else "RPAREN", value, None))
        else:
            word = match.group("word")
            upper = word.upper()
            if upper in {"AND", "OR", "NOT"}:
                tokens.append((upper, upper, None))
            elif word == "-":
                tokens.append(("NOT", "NOT", None))
            elif word.startswith("-") and len(word) > 1:
                tokens.append(("NOT", "NOT", None))
                tokens.append(("TERM", word[1:], None))
            else:
                tokens.append(("TERM", word, None))
        pos = match.end()
    return tokens


class _Parser:
    def __init__(self, tokens: Sequence[Tuple[str, str, Optional[str]]]):
        self.tokens = tokens
        self.index = 0

    def current(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else ("EOF", "", None)

    def accept(self, kind: str) -> bool:
        if self.current()[0] == kind:
            self.index += 1
            return True
        return False

    def parse(self) -> Optional[Node]:
        if not self.tokens:
            return None
        node = self.parse_or()
        if self.current()[0] != "EOF":
            kind, _, _ = self.current()
            raise QuerySyntaxError("Unexpected ')'" if kind == "RPAREN" else "Expected AND or OR")
        return node

    def parse_or(self) -> Node:
        nodes = [self.parse_and()]
        while self.accept("OR"):
            if self.current()[0] in {"EOF", "RPAREN", "AND", "OR"}:
                raise QuerySyntaxError("OR must have an expression on both sides")
            nodes.append(self.parse_and())
        return nodes[0] if len(nodes) == 1 else Node("OR", children=tuple(nodes))

    def parse_and(self) -> Node:
        nodes = [self.parse_unary()]
        while True:
            if self.accept("AND"):
                if self.current()[0] in {"EOF", "RPAREN", "AND", "OR"}:
                    raise QuerySyntaxError("AND must have an expression on both sides")
                nodes.append(self.parse_unary())
            elif self.current()[0] in {"TERM", "LPAREN", "NOT"}:
                # Adjacent terms mean AND, a convenient default for typed queries.
                nodes.append(self.parse_unary())
            else:
                break
        return nodes[0] if len(nodes) == 1 else Node("AND", children=tuple(nodes))

    def parse_unary(self) -> Node:
        if self.accept("NOT"):
            if self.current()[0] in {"EOF", "RPAREN", "AND", "OR"}:
                raise QuerySyntaxError("NOT must be followed by an expression")
            return Node("NOT", children=(self.parse_unary(),))
        if self.accept("LPAREN"):
            if self.current()[0] == "RPAREN":
                raise QuerySyntaxError("Empty parentheses are not allowed")
            node = self.parse_or()
            if not self.accept("RPAREN"):
                raise QuerySyntaxError("Missing closing ')' ")
            return node
        kind, value, field = self.current()
        if kind == "TERM":
            self.index += 1
            if not value:
                raise QuerySyntaxError("Search terms cannot be empty")
            return Node("TERM", value=value, field=field)
        if kind == "RPAREN":
            raise QuerySyntaxError("Unexpected ')' ")
        raise QuerySyntaxError("Expected a term, quote, or '(' ")


def parse_query(query: str) -> Optional[Node]:
    tree = _Parser(tokenize(query)).parse()

    def validate(node):
        if node is None:
            return
        if node.kind == "TERM" and node.field == "year":
            if not re.fullmatch(r"\d{4}(?:\.\.\d{4})?", node.value):
                raise QuerySyntaxError("year: expects YYYY or YYYY..YYYY")
            bounds = [int(part) for part in node.value.split("..")]
            if any(year < 1000 or year > 9999 for year in bounds):
                raise QuerySyntaxError("year: values must be between 1000 and 9999")
            if len(bounds) == 2 and bounds[0] > bounds[1]:
                raise QuerySyntaxError("year range start must not exceed end")
        if node.kind == "TERM" and node.field == "citations" and not re.fullmatch(r"(?:>=|<=|>|<|=)?\d+", node.value):
            raise QuerySyntaxError("citations: expects a number, >N, >=N, <N, or <=N")
        if node.kind == "TERM" and node.field == "open_access" and node.value.casefold() not in {"true", "false"}:
            raise QuerySyntaxError("open_access: expects true or false")
        if node.kind == "TERM" and node.field == "type" and node.value.casefold() not in ALLOWED_RECORD_TYPES:
            raise QuerySyntaxError("type: expects paper, person, organization, website, news, social, or other")
        if node.kind == "TERM" and node.field == "source":
            normalized = re.sub(r"[^a-z0-9]", "", node.value.casefold())
            if normalized not in ALLOWED_SOURCES:
                raise QuerySyntaxError("source: expects an available source name, academic, or all")
        if node.kind == "TERM" and node.field == "site":
            parsed_site = urlparse(node.value if "://" in node.value else "//" + node.value)
            host = parsed_site.hostname
            if (not host or parsed_site.path not in {"", "/"} or parsed_site.query or parsed_site.fragment
                    or parsed_site.username or parsed_site.password):
                raise QuerySyntaxError("site: expects a domain name, such as example.org")
        if node.kind == "TERM" and node.field == "language" and not re.fullmatch(
                r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", node.value):
            raise QuerySyntaxError("language: expects a language code such as en, fr, or zh-Hans")
        for child in node.children:
            validate(child)

    validate(tree)
    return tree


def positive_terms(node: Optional[Node], *, include_filter_values: bool = False, negated: bool = False) -> List[str]:
    terms: List[str] = []
    if node is None:
        return terms
    if node.kind == "NOT":
        return positive_terms(node.children[0], include_filter_values=include_filter_values, negated=not negated)
    if node.kind == "TERM":
        if not negated and (node.field is None or include_filter_values and node.field in {"title", "author"}):
            terms.append(node.value)
        return terms
    for child in node.children:
        terms.extend(positive_terms(child, include_filter_values=include_filter_values, negated=negated))
    return terms


def source_query(node: Optional[Node]) -> str:
    return " ".join(dict.fromkeys(positive_terms(node, include_filter_values=True)))


def source_query_variants(node: Optional[Node], max_variants: int = 8) -> List[str]:
    """Expand simple boolean expressions into bounded source-specific query branches."""
    def clauses(current: Optional[Node], negated=False):
        if current is None:
            return [[]]
        if current.kind == "NOT":
            return clauses(current.children[0], not negated)
        if current.kind == "TERM":
            searchable = current.field is None or current.field in {"title", "author"}
            return [[current.value] if searchable and not negated else []]

        effective_kind = current.kind
        if negated:
            effective_kind = "OR" if current.kind == "AND" else "AND"
        child_clauses = [clauses(child, negated) for child in current.children]
        if effective_kind == "OR":
            expanded = []
            for group in child_clauses:
                expanded.extend(group)
                if len(expanded) > max_variants:
                    raise QuerySyntaxError(f"Query expands to more than {max_variants} source searches; simplify its OR branches")
            return expanded
        combined = [[]]
        for group in child_clauses:
            combined = [left + right for left in combined for right in group]
            if len(combined) > max_variants:
                raise QuerySyntaxError(f"Query expands to more than {max_variants} source searches; simplify its OR branches")
        return combined

    variants = []
    all_clauses = clauses(node)
    if any(not clause for clause in all_clauses):
        raise QuerySyntaxError("Each OR branch must contain a positive search term; put NOT terms beside a positive term with AND")
    for clause in all_clauses:
        query = " ".join(dict.fromkeys(term for term in clause if term))
        if query and query not in variants:
            variants.append(query)
        if len(variants) >= max_variants:
            break

    def simple_words(current):
        if current is None:
            return []
        if current.kind == "TERM":
            if current.field is not None or not re.fullmatch(r"[\w]+", current.value, flags=re.UNICODE):
                return []
            return [current.value]
        if current.kind != "AND":
            return []
        words = []
        for child in current.children:
            child_words = simple_words(child)
            if not child_words:
                return []
            words.extend(child_words)
        return list(dict.fromkeys(words))

    # Broad retrieval searches add compact term combinations for plain text only.
    # Final result matching remains governed by the original query expression.
    words = simple_words(node)
    if len(words) >= 2 and len(variants) < max_variants:
        # Two-word queries need one-word retrievals as a recall fallback; longer
        # queries retain at least two thirds of the words per variant.
        combination_size = 1 if len(words) == 2 else max(2, (2 * len(words) + 2) // 3)
        if len(words) <= 8:
            candidates = list(combinations(words, combination_size))
        else:
            starts = sorted({
                round(index * (len(words) - combination_size) / max(1, max_variants - 2))
                for index in range(max_variants - 1)
            })
            windows = [tuple(words[start:start + combination_size]) for start in starts]
            edge = tuple(dict.fromkeys([words[0]] + words[-(combination_size - 1):]))
            candidates = windows + [edge]
        available = max_variants - len(variants)
        if len(candidates) > available:
            if available == 1:
                candidates = [candidates[len(candidates) // 2]]
            else:
                candidate_indexes = [round(index * (len(candidates) - 1) / (available - 1))
                                     for index in range(available)]
                candidates = [candidates[index] for index in candidate_indexes]
        for combination in candidates:
            query = " ".join(combination)
            if query not in variants:
                variants.append(query)
            if len(variants) >= max_variants:
                break

    # Search identifiers in a few provider-friendly forms. Some indexes drop
    # punctuation, while others require an exact phrase or split a code at the
    # letter/digit boundary. Final filtering still checks the original query.
    if node is not None and node.kind == "TERM" and node.field is None and len(variants) < max_variants:
        value = node.value.strip()
        compact = re.sub(r"[^A-Za-z0-9]", "", value)
        identifier_variants = []
        if _EMAIL_TERM_RE.fullmatch(value):
            identifier_variants = [value.split("@", 1)[0]]
        elif (8 <= len(compact) <= 40
              and len(re.findall(r"[A-Za-z]", compact)) <= 6
              and len(re.findall(r"\d", compact)) >= 4):
            boundary = re.search(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", compact)
            if boundary:
                identifier_variants.append(compact[:boundary.start()] + " " + compact[boundary.start():])
        for variant in identifier_variants:
            if variant not in variants and len(variants) < max_variants:
                variants.append(variant)
    return variants


def _year(record) -> Optional[int]:
    match = re.search(r"(?:19|20)\d{2}", str(getattr(record, "date", "") or ""))
    return int(match.group()) if match else None


def _site_matches(record, expected: str) -> bool:
    if "://" in expected:
        expected = urlparse(expected).hostname or ""
    expected = expected.removeprefix("*.")
    host = urlparse(getattr(record, "url", "") or "").hostname or ""
    host, expected = host.casefold().removeprefix("www."), expected.casefold().strip().removeprefix("www.")
    return host == expected or host.endswith("." + expected)


def normalize_search_text(value: str) -> str:
    """Normalize Unicode spelling, accents, Arabic diacritics, and punctuation for matching."""
    value = unicodedata.normalize("NFKD", str(value or "").casefold()).translate(ARABIC_FOLDS)
    normalized = []
    for char in value:
        if unicodedata.combining(char):
            continue
        normalized.append(char if char.isalnum() or char == "_" else " ")
    return re.sub(r"\s+", " ", "".join(normalized)).strip()


def _identifier_pattern(value: str):
    compact = re.sub(r"[^A-Za-z0-9]", "", str(value or ""))
    letters = len(re.findall(r"[A-Za-z]", compact))
    digits = len(re.findall(r"\d", compact))
    if not 8 <= len(compact) <= 40 or letters > 6 or digits < 4:
        return None
    pattern = r"(?<![A-Za-z0-9])" + r"[\s._:/-]{0,3}".join(
        re.escape(char) for char in compact
    ) + r"(?![A-Za-z0-9])"
    return re.compile(pattern, re.IGNORECASE)


def normalized_text_contains(text: str, value: str) -> bool:
    # Identifiers are often displayed with punctuation or grouping whitespace
    # that is absent from the user's query (for example AB1234567 / AB-123 4567).
    identifier_pattern = _identifier_pattern(value)
    if identifier_pattern:
        return bool(identifier_pattern.search(unquote(str(text or ""))))

    if _EMAIL_TERM_RE.fullmatch(str(value or "").strip()):
        pattern = r"(?<![A-Za-z0-9._%+-])" + re.escape(value.strip())
        pattern += r"(?![A-Za-z0-9.-])"
        return bool(re.search(pattern, unquote(str(text or "")), re.IGNORECASE))

    normalized_text = normalize_search_text(text)
    normalized_value = normalize_search_text(value)
    if not normalized_text or not normalized_value:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(normalized_value) + r"(?!\w)", normalized_text, re.UNICODE))


def _normalized_contains(text: str, value: str) -> bool:
    """Backward-compatible internal alias for normalized token matching."""
    return normalized_text_contains(text, value)


def _field_matches(record, field: str, value: str) -> Optional[bool]:
    value = value.strip()
    if field == "title":
        title = getattr(record, "title", "") or ""
        return _normalized_contains(title, value) if title else None
    if field == "author":
        authors = getattr(record, "authors", "") or ""
        return _normalized_contains(authors, value) if authors else None
    if field == "site":
        if not (urlparse(getattr(record, "url", "") or "").hostname):
            return None
        return _site_matches(record, value)
    if field == "type":
        kind = getattr(record, "kind", "unknown") or "unknown"
        return None if kind.casefold() == "unknown" else value.casefold() == kind.casefold()
    if field == "source":
        sources = getattr(record, "sources", None) or [getattr(record, "source", "")]
        sources = [source for source in sources if source]
        if not sources:
            return None
        normalize = lambda text: re.sub(r"[^a-z0-9]", "", text.casefold())
        if normalize(value) == "all":
            return True
        if normalize(value) == "academic":
            return any(normalize(source) not in {"duckduckgo", "google"} for source in sources)
        return any(normalize(value) in normalize(source) for source in sources)
    if field == "year":
        year = _year(record)
        if year is None:
            return None
        range_match = re.fullmatch(r"(\d{4})\.\.(\d{4})", value)
        if range_match:
            start, end = map(int, range_match.groups())
            if start > end:
                raise QuerySyntaxError("Year range start must not exceed end")
            return start <= year <= end
        if re.fullmatch(r"\d{4}", value):
            return year == int(value)
        raise QuerySyntaxError("year: expects YYYY or YYYY..YYYY")
    if field == "citations":
        citation_count = getattr(record, "citations", None)
        match = re.fullmatch(r"(>=|<=|>|<|=)?(\d+)", value)
        if not match:
            raise QuerySyntaxError("citations: expects a number, >N, >=N, <N, or <=N")
        if citation_count is None:
            return None
        operator, raw_number = match.groups()
        number = int(raw_number)
        return {None: citation_count == number, "=": citation_count == number, ">": citation_count > number,
                ">=": citation_count >= number, "<": citation_count < number, "<=": citation_count <= number}[operator]
    if field == "language":
        language = getattr(record, "language", None)
        return None if not language else language.casefold() == value.casefold()
    if field == "open_access":
        if value.casefold() not in {"true", "false"}:
            raise QuerySyntaxError("open_access: expects true or false")
        open_access = getattr(record, "open_access", None)
        return None if open_access is None else open_access is (value.casefold() == "true")
    raise QuerySyntaxError(f"Unsupported search field '{field}'")


def unknown_query_fields(record, node: Optional[Node], negated: bool = False) -> List[str]:
    """Return positive metadata fields with missing values, for visible uncertainty notes."""
    if node is None:
        return []
    if node.kind == "NOT":
        return unknown_query_fields(record, node.children[0], not negated)
    if node.kind == "TERM":
        if not negated and node.field and _field_matches(record, node.field, node.value) is None:
            return [node.field]
        return []
    fields = []
    for child in node.children:
        fields.extend(unknown_query_fields(record, child, negated))
    return list(dict.fromkeys(fields))


def matches_query(record, node: Optional[Node], *, include_unknown_fields: bool = False) -> Optional[bool]:
    if node is None:
        return True
    if node.kind == "AND":
        values = [matches_query(record, child, include_unknown_fields=include_unknown_fields) for child in node.children]
        if False in values:
            return False
        return None if None in values else True
    if node.kind == "OR":
        values = [matches_query(record, child, include_unknown_fields=include_unknown_fields) for child in node.children]
        if True in values:
            return True
        return None if None in values else False
    if node.kind == "NOT":
        value = matches_query(record, node.children[0], include_unknown_fields=include_unknown_fields)
        return None if value is None else not value
    if node.field:
        value = _field_matches(record, node.field, node.value)
        # Scope selectors must remain fail-closed even when optional metadata is included.
        return True if value is None and include_unknown_fields and node.field not in {"site", "source"} else value
    text = " ".join((
        getattr(record, "title", "") or "", getattr(record, "abstract", "") or "",
        getattr(record, "authors", "") or "", getattr(record, "url", "") or "",
        unquote(getattr(record, "url", "") or ""), getattr(record, "doi", "") or "",
        " ".join(getattr(record, "deep_context", []) or []),
    ))
    return _normalized_contains(text, node.value) if text else None


def evaluate_query(query: str, record) -> bool:
    return matches_query(record, parse_query(query)) is True
