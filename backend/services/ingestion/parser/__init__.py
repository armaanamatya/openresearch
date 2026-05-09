"""Parser abstraction + parsed-paper artifact model (#13)."""

from backend.services.ingestion.parser.aggregate import (
    ParsedPaperAggregate,
    ParsedPaperState,
)
from backend.services.ingestion.parser.events import (
    FigureExtracted,
    ParsingCompleted,
    ParsingFailed,
    ParsingStarted,
    ReferenceExtracted,
    SectionExtracted,
)
from backend.services.ingestion.parser.interface import Parser, ParseError
from backend.services.ingestion.parser.model import (
    Figure,
    Reference,
    Section,
    section_id_for,
    reference_id_for,
)
from backend.services.ingestion.parser.service import (
    ParserAppService,
    ParserError,
    StartParsing,
)

__all__ = [
    "Figure",
    "FigureExtracted",
    "Parser",
    "ParseError",
    "ParserAppService",
    "ParserError",
    "ParsedPaperAggregate",
    "ParsedPaperState",
    "ParsingCompleted",
    "ParsingFailed",
    "ParsingStarted",
    "Reference",
    "ReferenceExtracted",
    "Section",
    "SectionExtracted",
    "StartParsing",
    "reference_id_for",
    "section_id_for",
]
