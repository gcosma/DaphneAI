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
    source_document: str
    rec_id: Optional[str] = None  # raw label, e.g. "1", "2018/007"
    rec_number: Optional[int] = None  # numeric id when applicable
    rec_type: Optional[str] = None  # e.g. "numbered", "action_verb"
    detection_method: Optional[str] = None  # e.g. "heading", "verb_based"


@dataclass
class Response:
    """
    A single extracted response in the v2 pipeline.
    """

    text: str
    span: Span
    source_document: str
    response_type: str  # e.g. "structured", "action_verb"
    rec_id: Optional[str] = None
    rec_number: Optional[int] = None


@dataclass
class AlignmentResult:
    """
    Alignment between a recommendation and the best matching response(s).
    """

    recommendation: Recommendation
    response: Optional[Response]
    similarity: Optional[float]
    match_method: str  # e.g. "number_first", "semantic", "keyword"
