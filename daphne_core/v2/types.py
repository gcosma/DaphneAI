from dataclasses import dataclass
from typing import List, Optional, Tuple


Span = Tuple[int, int]


@dataclass
class PreprocessedText:
    """
    Normalised text plus structural spans used by v2 components.

    All spans are (start, end) offsets into `text`.
    """

    text: str
    sentence_spans: List[Span]
    page_spans: List[Span]


@dataclass
class Recommendation:
    """
    A single extracted recommendation in the v2 pipeline.
    """

    text: str
    span: Span
    rec_number: Optional[int]
    source_document: str


@dataclass
class Response:
    """
    A single extracted response in the v2 pipeline.
    """

    text: str
    span: Span
    rec_number: Optional[int]
    source_document: str
    response_type: str  # e.g. "structured", "scattered"


@dataclass
class AlignmentResult:
    """
    Alignment between a recommendation and the best matching response(s).
    """

    recommendation: Recommendation
    response: Optional[Response]
    similarity: Optional[float]
    match_method: str  # e.g. "number_first", "semantic", "keyword"

