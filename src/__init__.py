"""
PHAT - Patent Prosecution History Analysis Tool
v2.1 - Added Analyst Layer (Post-Processing & Synthesis)
"""

from .pdf_processor import PDFProcessor, DocumentSection
from .analysis_engine import AnalysisEngine
from .database import (
    DatabaseManager, ValidityRisk, TermSynthesis, ClaimNarrative
)
from .ai_providers import AIProviderFactory, ModelSelector, TaskType
from .report_generator import ReportGenerator
from .chunking import DocumentChunker, ResultMerger
from .ocr_cleaner import create_ocr_cleaner
from .verification import create_content_verifier
from .final_claims import parse_google_patents_claims

__version__ = "2.1.0"
__all__ = [
    "PDFProcessor",
    "DocumentSection",
    "AnalysisEngine",
    "DatabaseManager",
    "AIProviderFactory",
    "ModelSelector",
    "TaskType",
    "ReportGenerator",
    "DocumentChunker",
    "ResultMerger",
    "ValidityRisk",
    "TermSynthesis",
    "ClaimNarrative",
]
