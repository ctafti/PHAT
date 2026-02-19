"""
PHAT PDF Processor v2.0
Handles PDF text extraction, bookmark-based segmentation, and document classification.

Key change: Uses PDF bookmarks as PRIMARY segmentation strategy.
USPTO Image File Wrapper PDFs have reliable bookmarks that define document boundaries.
Pattern-based segmentation is the FALLBACK when bookmarks are absent.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

import pdfplumber
from pypdf import PdfReader

try:
    from .ai_providers import AIProvider
except ImportError:
    AIProvider = Any

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Represents a section of the patent file history"""
    document_type: str
    page_start: int
    page_end: int
    text: str
    date: Optional[datetime] = None
    mail_date: Optional[datetime] = None
    is_high_priority: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedClaim:
    """Represents an extracted claim from the document"""
    number: int
    text: str
    is_independent: bool
    depends_on: Optional[int] = None


class PDFProcessor:
    """
    Processes patent file history PDFs.
    
    Segmentation strategy (in order):
    1. BOOKMARKS (primary) - USPTO PDFs have reliable bookmark outlines
    2. PATTERNS (fallback)  - Regex patterns when bookmarks are missing
    3. SINGLE SECTION       - Last resort: treat entire PDF as one document
    """
    
    _verbose = False
    
    @classmethod
    def set_verbose(cls, verbose: bool):
        cls._verbose = verbose
    
    def _log_verbose(self, message: str, data: Any = None):
        if self._verbose:
            logger.info(f"[VERBOSE] {message}")
            if data is not None:
                data_str = str(data)
                if len(data_str) > 2000:
                    logger.info(f"[VERBOSE] Data (truncated {len(data_str)} chars):\n{data_str[:1000]}\n...[TRUNCATED]...\n{data_str[-1000:]}")
                else:
                    logger.info(f"[VERBOSE] Data:\n{data_str}")

    # =========================================================================
    # Document type patterns for classification
    # =========================================================================
    DOCUMENT_PATTERNS = {
        "Office Action": [
            r"office\s+action",
            r"non-?final\s+rejection",
            r"final\s+rejection",
            r"first\s+office\s+action",
            r"UNITED STATES PATENT AND TRADEMARK OFFICE.*Office Action",
        ],
        "Amendment": [
            r"amendment",
            r"response\s+to\s+office\s+action",
            r"reply\s+to\s+office\s+action",
            r"applicant.?s?\s+response",
        ],
        "Appeal Brief": [
            r"appeal\s+brief",
            r"appellant.?s?\s+brief",
        ],
        "Reply Brief": [
            r"reply\s+brief",
        ],
        "Notice of Allowance": [
            r"notice\s+of\s+allowance",
            r"allowance\s+notice",
        ],
        "Restriction Requirement": [
            r"restriction\s+requirement",
            r"election\s+requirement",
            r"requirement\s+for\s+restriction",
        ],
        "Interview Summary": [
            r"interview\s+summary",
            r"examiner\s+interview\s+summary",
        ],
        "Information Disclosure Statement": [
            r"information\s+disclosure\s+statement",
            r"\bIDS\b",
        ],
        "Claims": [
            r"^claims?\s*$",
            r"what\s+is\s+claimed\s+is",
        ],
        "Specification": [
            r"specification",
            r"description\s+of.*embodiments?",
        ],
        "Terminal Disclaimer": [
            r"terminal\s+disclaimer",
        ],
        "PTAB Decision": [
            r"board\s+decision",
            r"PTAB\s+decision",
            r"patent\s+trial\s+and\s+appeal\s+board",
        ],
        "Fee Transmittal": [
            r"fee\s+transmittal",
            r"fee\s+worksheet",
        ],
        "Power of Attorney": [
            r"power\s+of\s+attorney",
        ],
        "Application Data Sheet": [
            r"application\s+data\s+sheet",
        ],
        "Filing Receipt": [
            r"filing\s+receipt",
        ],
        "Reasons for Allowance": [
            r"reasons?\s+for\s+allowance",
            r"statement\s+of\s+reasons?\s+for\s+allowance",
        ],
    }
    
    SKIP_DOCUMENT_TYPES = [
        "Fee Transmittal",
        "Power of Attorney",
        "Application Data Sheet",
        "Filing Receipt",
        "Administrative",
        "Drawings",
        "Transmittal",
        "Examiner Search",
        "Citation List",
    ]
    
    HIGH_PRIORITY_TYPES = [
        "Appeal Brief",
        "Reply Brief",
        "Notice of Allowance",
        "Reasons for Allowance",
        "PTAB Decision",
    ]
    
    def __init__(self, pdf_path: str, ai_provider: Optional[AIProvider] = None):
        self.pdf_path = Path(pdf_path)
        self.ai_provider = ai_provider
        self.text_by_page: List[str] = []
        self.bookmarks: List[Dict] = []
        self.total_pages: int = 0
        self.sections: List[DocumentSection] = []

    # =========================================================================
    # TEXT EXTRACTION
    # =========================================================================
    
    def extract_text(self) -> List[str]:
        """Extract text from all pages of the PDF using pdfplumber"""
        logger.info(f"Extracting text from {self.pdf_path}")
        self._log_verbose(f"Starting PDF text extraction from: {self.pdf_path}")
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                self.total_pages = len(pdf.pages)
                self.text_by_page = []
                self._log_verbose(f"PDF opened. Total pages: {self.total_pages}")
                
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text() or ""
                        self.text_by_page.append(text)
                        logger.debug(f"Extracted page {i+1}/{self.total_pages}")
                    except Exception as e:
                        logger.warning(f"Failed to extract page {i+1}: {e}")
                        self.text_by_page.append("")
                        
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise
            
        logger.info(f"Extracted {len(self.text_by_page)} pages ({sum(len(t) for t in self.text_by_page)} chars total)")
        return self.text_by_page

    # =========================================================================
    # BOOKMARK EXTRACTION (PRIMARY STRATEGY)
    # =========================================================================
    
    def extract_bookmarks(self) -> List[Dict]:
        """Extract bookmarks/outline from PDF using pypdf"""
        self.bookmarks = []
        try:
            reader = PdfReader(self.pdf_path)
            if reader.outline:
                self._parse_outline(reader, reader.outline)
                logger.info(f"Extracted {len(self.bookmarks)} bookmarks from PDF")
                self._log_verbose("Bookmarks:", self.bookmarks)
            else:
                logger.info("No bookmarks/outline found in PDF")
        except Exception as e:
            logger.warning(f"Failed to extract bookmarks: {e}")
        
        return self.bookmarks
    
    def _parse_outline(self, reader: PdfReader, outline, level: int = 0):
        """Recursively parse PDF outline/bookmarks with robust page resolution"""
        for item in outline:
            if isinstance(item, list):
                self._parse_outline(reader, item, level + 1)
            else:
                try:
                    title = item.title if hasattr(item, 'title') else str(item)
                    page_num = self._resolve_bookmark_page(reader, item)
                    
                    self.bookmarks.append({
                        'title': title,
                        'page': page_num,
                        'level': level,
                    })
                except Exception as e:
                    logger.debug(f"Failed to parse bookmark item: {e}")
    
    def _resolve_bookmark_page(self, reader: PdfReader, bookmark) -> Optional[int]:
        """
        Resolve a bookmark to a 0-based page number.
        Handles IndirectObject references and different pypdf versions.
        """
        try:
            # Method 1: Direct page attribute (destination)
            if hasattr(bookmark, 'page') and bookmark.page is not None:
                page_obj = bookmark.page
                # If it's an IndirectObject, resolve it
                if hasattr(page_obj, 'get_object'):
                    page_obj = page_obj.get_object()
                # Search pages list for matching object
                for i, page in enumerate(reader.pages):
                    if page.get_object() == page_obj:
                        return i
                # Fallback: if page is already an int
                if isinstance(page_obj, int):
                    return page_obj
        except Exception:
            pass
        
        try:
            # Method 2: Get destination from reader
            dest = reader.get_destination_page_number(bookmark)
            if dest is not None:
                return dest
        except Exception:
            pass
        
        return None
    
    def _classify_bookmark_title(self, title: str) -> str:
        """
        Classify a bookmark title string into a document type.
        USPTO bookmarks often have descriptive titles like:
        "Non-Final Rejection (21 August 2008)", "Response to Office Action", etc.
        
        NOTE: USPTO bookmark titles often include dates in parentheses.
        We strip those before matching.
        """
        # Strip trailing date patterns like "(26 June 2007)" before matching
        import re as _re
        title_clean = _re.sub(r'\s*\(\d{1,2}\s+\w+\s+\d{4}\)\s*$', '', title)
        title_lower = title_clean.lower().strip()
        
        # Direct bookmark title → document type mappings (most common USPTO bookmark titles)
        bookmark_mappings = {
            # Office Actions
            "non-final rejection": "Office Action",
            "non-final office action": "Office Action",
            "final rejection": "Office Action",
            "final office action": "Office Action",
            "office action summary": "Office Action",
            "office action": "Office Action",
            "examiner's answer": "Office Action",
            "advisory action": "Office Action",
            # Amendments / Responses
            "amendment": "Amendment",
            "response to office action": "Amendment",
            "reply to office action": "Amendment",
            "applicant arguments/remarks": "Amendment",
            "applicant's response": "Amendment",
            "response after final": "Amendment",
            "amendment after final": "Amendment",
            "remarks": "Amendment",
            "amendment/request for reconsideration": "Amendment",
            "supplemental response": "Amendment",
            "supplemental amendment": "Amendment",
            "amendment submitted/entered with filing of continued prosecution application": "Amendment",
            # Allowance
            "notice of allowance": "Notice of Allowance",
            "notice of allowability": "Notice of Allowance",
            "reasons for allowance": "Reasons for Allowance",
            "statement of reasons for allowance": "Reasons for Allowance",
            "examiner's reasons for allowance": "Reasons for Allowance",
            # Restriction
            "restriction requirement": "Restriction Requirement",
            "election/restriction": "Restriction Requirement",
            "requirement for restriction/election": "Restriction Requirement",
            # Interview
            "interview summary": "Interview Summary",
            "examiner interview summary": "Interview Summary",
            "examiner interview summary record": "Interview Summary",
            "applicant summary of interview with examiner": "Interview Summary",
            "letter requesting interview with examiner": "Interview Summary",
            # Appeals
            "appeal brief": "Appeal Brief",
            "reply brief": "Reply Brief",
            # Terminal Disclaimer
            "terminal disclaimer": "Terminal Disclaimer",
            # PTAB
            "decision on appeal": "PTAB Decision",
            "board decision": "PTAB Decision",
            # Claims
            "claims": "Claims",
            # IDS
            "information disclosure statement": "Information Disclosure Statement",
            "ids": "Information Disclosure Statement",
            # Specification
            "specification": "Specification",
            "abstract": "Specification",
            # RCE
            "request for continued examination": "Request for Continued Examination",
            # Boilerplate / Administrative (these get skipped in analysis)
            "fee transmittal": "Fee Transmittal",
            "fee worksheet": "Fee Transmittal",
            "power of attorney": "Power of Attorney",
            "application data sheet": "Application Data Sheet",
            "filing receipt": "Filing Receipt",
            "drawings": "Drawings",
            "transmittal of new application": "Transmittal",
            "transmittal letter": "Transmittal",
            "electronic filing system acknowledgment receipt": "Filing Receipt",
            "authorization for extension of time": "Administrative",
            "extension of time": "Administrative",
            "pre-exam formalities notice": "Administrative",
            "applicant response to pre-exam formalities notice": "Administrative",
            "oath or declaration": "Administrative",
            "notice of publication": "Administrative",
            "examiner's search strategy and results": "Examiner Search",
            "list of references cited by examiner": "Citation List",
            "list of references cited by applicant": "Citation List",
            "index of claims": "Administrative",
            "search information including classification": "Examiner Search",
            "bibliographic data sheet": "Administrative",
            "issue information": "Administrative",
            "issue fee payment": "Administrative",
            "issue notification": "Administrative",
            "communication - re": "Administrative",
            "assignee showing of ownership": "Administrative",
            "miscellaneous incoming letter": "Correspondence",
            "miscellaneous internal document": "Administrative",
            "documents submitted with 371": "Administrative",
            "certification of micro entity": "Administrative",
        }
        
        # Check substring matches — longest patterns first to prevent
        # e.g. "claims" matching before "index of claims"
        for pattern, doc_type in sorted(bookmark_mappings.items(), key=lambda x: -len(x[0])):
            if pattern in title_lower:
                return doc_type
        
        # Fallback to regex pattern matching on the title text
        result = self.classify_document_type(title_clean)
        if result != "Unknown":
            return result
        
        # Final fallback: label as Correspondence rather than calling AI
        # This avoids flooding the API with classification calls for
        # administrative documents that will likely be skipped anyway
        logger.debug(f"Bookmark title not recognized, defaulting to Correspondence: '{title}'")
        return "Correspondence"

    # =========================================================================
    # DOCUMENT SEGMENTATION
    # =========================================================================
    
    def segment_into_sections(self) -> List[DocumentSection]:
        """
        Segment the PDF into logical document sections.
        
        Strategy:
        1. Try bookmarks first (USPTO PDFs have reliable outlines)
        2. Fall back to pattern-based detection
        3. Last resort: single-section fallback
        """
        self._log_verbose("Starting PDF segmentation")
        
        if not self.text_by_page:
            self.extract_text()
        
        # Strategy 1: Bookmark-based segmentation (PRIMARY)
        self.extract_bookmarks()
        
        if self.bookmarks:
            sections = self._segment_by_bookmarks()
            if sections:
                self.sections = sections
                logger.info(f"Bookmark segmentation: {len(self.sections)} sections")
                self._log_verbose(f"Section types: {[s.document_type for s in self.sections]}")
                return self.sections
            logger.warning("Bookmarks found but segmentation failed, falling back to patterns")
        
        # Strategy 2: Pattern-based segmentation (FALLBACK)
        boundaries = self._detect_pattern_boundaries()
        self._log_verbose(f"Pattern detection found {len(boundaries)} boundaries", boundaries[:20])
        
        if boundaries:
            self.sections = self._build_sections_from_boundaries(boundaries)
            logger.info(f"Pattern segmentation: {len(self.sections)} sections")
            return self.sections
        
        # Strategy 3: Single-section fallback (LAST RESORT)
        logger.warning("No boundaries detected - treating entire PDF as single section")
        full_text = "\n".join(self.text_by_page)
        doc_type = self.classify_document_type(full_text)
        
        if doc_type == "Unknown" and self.ai_provider:
            doc_type = self._classify_with_ai(full_text[:1500])
        
        self.sections = [DocumentSection(
            document_type=doc_type,
            page_start=0,
            page_end=len(self.text_by_page) - 1,
            text=full_text,
            date=self.extract_date(full_text),
            is_high_priority=doc_type in self.HIGH_PRIORITY_TYPES,
        )]
        return self.sections
    
    def _segment_by_bookmarks(self) -> List[DocumentSection]:
        """
        Segment PDF using bookmark boundaries.
        Each top-level bookmark defines a document section boundary.
        """
        # Filter to bookmarks that have resolved page numbers
        valid_bookmarks = [b for b in self.bookmarks if b['page'] is not None]
        
        if not valid_bookmarks:
            logger.warning("No bookmarks with valid page numbers")
            return []
        
        # Sort by page number
        valid_bookmarks.sort(key=lambda b: b['page'])
        
        # Use only top-level bookmarks (level 0) for section boundaries
        # If all bookmarks are level > 0, use all of them
        top_level = [b for b in valid_bookmarks if b['level'] == 0]
        if not top_level:
            top_level = valid_bookmarks
        
        sections = []
        for i, bookmark in enumerate(top_level):
            start_page = bookmark['page']
            
            # End page is one before the next bookmark's page, or last page
            if i + 1 < len(top_level):
                end_page = top_level[i + 1]['page'] - 1
            else:
                end_page = len(self.text_by_page) - 1
            
            # Ensure valid range
            end_page = max(start_page, min(end_page, len(self.text_by_page) - 1))
            
            # Combine text for this section
            section_text = "\n".join(self.text_by_page[start_page:end_page + 1])
            
            # Classify document type from bookmark title, with text fallback
            doc_type = self._classify_bookmark_title(bookmark['title'])
            if doc_type == "Unknown":
                # Try classifying from the section text itself
                doc_type = self.classify_document_type(section_text[:3000])
            if doc_type == "Unknown" and self.ai_provider:
                doc_type = self._classify_with_ai(section_text[:1500])
            
            section = DocumentSection(
                document_type=doc_type,
                page_start=start_page,
                page_end=end_page,
                text=section_text,
                # Fix #8: Try bookmark title date first, then fall back to body text
                date=self._extract_date_from_bookmark_title(bookmark['title']) or self.extract_date(section_text),
                is_high_priority=doc_type in self.HIGH_PRIORITY_TYPES,
                metadata={'bookmark_title': bookmark['title']},
            )
            sections.append(section)
            self._log_verbose(
                f"Bookmark section: '{bookmark['title']}' -> {doc_type} "
                f"(pages {start_page}-{end_page}, {len(section_text)} chars)"
            )
        
        return sections
    
    def _detect_pattern_boundaries(self) -> List[Tuple[int, str]]:
        """Detect document boundaries by scanning pages for header patterns"""
        boundaries = []
        
        section_patterns = [
            r"^\s*UNITED\s+STATES\s+PATENT\s+AND\s+TRADEMARK\s+OFFICE",
            r"^\s*IN\s+THE\s+UNITED\s+STATES\s+PATENT\s+AND\s+TRADEMARK\s+OFFICE",
            r"^\s*AMENDMENT",
            r"^\s*REMARKS",
            r"^\s*CLAIMS?",
            r"^\s*RESPONSE\s+TO\s+OFFICE\s+ACTION",
            r"^\s*NOTICE\s+OF\s+ALLOWANCE",
            r"^\s*OFFICE\s+ACTION\s+SUMMARY",
            r"^\s*APPLICATION\s+NUMBER",
            r"^\s*APPEAL\s+BRIEF",
            r"^\s*RESTRICTION\s+REQUIREMENT",
        ]
        
        for i, text in enumerate(self.text_by_page):
            for pattern in section_patterns:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    doc_type = self.classify_document_type(text)
                    boundaries.append((i, doc_type))
                    break
        
        return boundaries
    
    def _build_sections_from_boundaries(self, boundaries: List[Tuple[int, str]]) -> List[DocumentSection]:
        """Build DocumentSection objects from detected boundaries"""
        sections = []
        
        for i, (start_page, doc_type) in enumerate(boundaries):
            if i + 1 < len(boundaries):
                end_page = boundaries[i + 1][0] - 1
            else:
                end_page = len(self.text_by_page) - 1
            
            section_text = "\n".join(self.text_by_page[start_page:end_page + 1])
            
            # AI fallback for Unknown types
            if doc_type == "Unknown" and self.ai_provider:
                doc_type = self._classify_with_ai(section_text[:1500])
            
            section = DocumentSection(
                document_type=doc_type,
                page_start=start_page,
                page_end=end_page,
                text=section_text,
                date=self.extract_date(section_text),
                is_high_priority=doc_type in self.HIGH_PRIORITY_TYPES,
            )
            sections.append(section)
            self._log_verbose(f"Pattern section {i+1}: {doc_type} (pages {start_page}-{end_page})")
        
        return sections

    # =========================================================================
    # DOCUMENT CLASSIFICATION
    # =========================================================================
    
    def classify_document_type(self, text: str) -> str:
        """Classify document type based on text content. Prioritizes substantive over boilerplate."""
        text_lower = text.lower()
        
        matches = []
        for doc_type, patterns in self.DOCUMENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matches.append(doc_type)
                    break
        
        if not matches:
            return "Unknown"
        
        if len(matches) == 1:
            return matches[0]
        
        # Prioritize substantive types over boilerplate
        substantive = [m for m in matches if m not in self.SKIP_DOCUMENT_TYPES]
        if substantive:
            for t in substantive:
                if t in self.HIGH_PRIORITY_TYPES:
                    return t
            return substantive[0]
        
        return matches[0]
    
    def _classify_with_ai(self, text_snippet: str) -> str:
        """Use AI to classify document type when regex fails"""
        if not self.ai_provider:
            return "Unknown"
        
        try:
            prompt = (
                "Identify the type of this patent document based on the text below. "
                "Return ONLY the document type name from this list: Office Action, Amendment, "
                "Notice of Allowance, Appeal Brief, Reply Brief, Interview Summary, "
                "Restriction Requirement, Advisory Action, Examiner's Amendment, "
                "Terminal Disclaimer, Claims, Reasons for Allowance. "
                "If none fit, return 'Correspondence'.\n\n"
                f"TEXT:\n{text_snippet[:1000]}"
            )
            doc_type = self.ai_provider.complete(prompt, system_prompt="You are a classifier.").strip().strip('"\'')
            return doc_type
        except Exception as e:
            logger.warning(f"AI classification failed: {e}")
            return "Unknown"

    # =========================================================================
    # DATE EXTRACTION
    # =========================================================================
    
    def _extract_date_from_bookmark_title(self, title: str) -> Optional[datetime]:
        """Extract date from bookmark title like 'Claims (26 June 2007)'
        
        Gap Analysis Fix #8: Bookmark titles contain reliable dates that were
        being stripped but never captured. This fixes 'Unknown' dates for
        Claims, Specification, Notice of Allowance, etc.
        """
        match = re.search(r'\((\d{1,2}\s+\w+\s+\d{4})\)\s*$', title)
        if match:
            date_str = match.group(1)
            for fmt in ["%d %B %Y", "%d %b %Y"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
        return None
    
    def extract_date(self, text: str) -> Optional[datetime]:
        """Extract date focusing on USPTO headers first"""
        # Priority 1: "Date Mailed:" / "Mailed" (standard USPTO)
        priority_patterns = [
            r"(?:Date Mailed|Mailed|DATE MAILED):?\s*(\d{1,2}/\d{1,2}/\d{2,4})",
            r"(?:Date Mailed|Mailed|DATE MAILED):?\s*([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
        ]
        
        for pattern in priority_patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE)
            if match:
                parsed = self._try_parse_date(match.group(1))
                if parsed:
                    return parsed
        
        # Priority 2: General date patterns
        general_patterns = [
            r"(?:Date|Mailed|Filed|dated)[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})",
            r"(\d{1,2}/\d{1,2}/\d{2,4})",
            r"(\w+\s+\d{1,2},?\s+\d{4})",
            r"(\d{4}-\d{2}-\d{2})",
        ]
        
        for pattern in general_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                parsed = self._try_parse_date(match.group(1))
                if parsed:
                    return parsed
        
        return None
    
    def _try_parse_date(self, date_str: str) -> Optional[datetime]:
        """Try to parse a date string with multiple formats"""
        for fmt in ["%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%B %d %Y", "%Y-%m-%d"]:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None

    # =========================================================================
    # CLAIMS EXTRACTION
    # =========================================================================
    
    def extract_claims_from_text(self, text: str) -> List[ExtractedClaim]:
        """Extract individual claims from claims section text"""
        claims = []
        
        claim_pattern = r"(?:^|\n)\s*(\d+)\.\s+(.+?)(?=\n\s*\d+\.|$)"
        matches = re.findall(claim_pattern, text, re.DOTALL)
        
        for num_str, claim_text in matches:
            num = int(num_str)
            claim_text = claim_text.strip()
            
            depends_on = None
            is_independent = True
            
            dep_patterns = [
                r"claim\s+(\d+)",
                r"according\s+to\s+claim\s+(\d+)",
                r"as\s+set\s+forth\s+in\s+claim\s+(\d+)",
            ]
            
            for pattern in dep_patterns:
                match = re.search(pattern, claim_text, re.IGNORECASE)
                if match:
                    depends_on = int(match.group(1))
                    is_independent = False
                    break
            
            claims.append(ExtractedClaim(
                number=num,
                text=claim_text,
                is_independent=is_independent,
                depends_on=depends_on,
            ))
        
        return claims

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_relevant_sections(self, skip_types: Optional[List[str]] = None) -> List[DocumentSection]:
        """Get sections relevant for analysis (excluding boilerplate)"""
        if not self.sections:
            self.segment_into_sections()
        skip = skip_types or self.SKIP_DOCUMENT_TYPES
        return [s for s in self.sections if s.document_type not in skip]
    
    def get_full_text(self) -> str:
        """Get full text of the entire PDF"""
        if not self.text_by_page:
            self.extract_text()
        return "\n".join(self.text_by_page)
    
    def get_patent_info(self) -> Dict[str, Any]:
        """Extract basic patent information from the PDF"""
        full_text = self.get_full_text()[:10000]
        
        info = {
            'patent_number': None,
            'application_number': None,
            'title': None,
            'filing_date': None,
            'assignee': None,
            'inventors': [],
        }
        
        # Patent number
        patent_match = re.search(
            r"(?:Patent\s+No\.?|US)\s*:?\s*(\d{1,2},?\d{3},?\d{3})", full_text, re.IGNORECASE
        )
        if patent_match:
            info['patent_number'] = patent_match.group(1).replace(",", "")
        
        # Application number
        app_match = re.search(
            r"(?:Application\s+No\.?|App\.?\s+No\.?|Serial\s+No\.?)\s*:?\s*(\d{2}/[\d,]+)",
            full_text, re.IGNORECASE,
        )
        if app_match:
            info['application_number'] = app_match.group(1).strip(" ,.")
        
        # Title — Fix #11: Exclude common false matches like "Patents", "Patent and Trademark"
        TITLE_FALSE_POSITIVES = {
            'patents', 'patent and trademark', 'patent and trademark office',
            'united states patent and trademark office', 'patent application',
            'patent number', 'patent no', 'patent',
        }
        title_match = re.search(r"(?:Title|FOR)\s*:?\s*([^\n]+)", full_text, re.IGNORECASE)
        if title_match:
            candidate = title_match.group(1).strip()
            if candidate.lower().rstrip('.') not in TITLE_FALSE_POSITIVES and len(candidate) > 5:
                info['title'] = candidate
        
        # If title extraction failed, try the specification section
        if not info['title']:
            spec_title_match = re.search(
                r"(?:TITLE\s+OF\s+(?:THE\s+)?INVENTION|FIELD\s+OF\s+(?:THE\s+)?INVENTION)\s*:?\s*([^\n]+)",
                full_text, re.IGNORECASE
            )
            if spec_title_match:
                candidate = spec_title_match.group(1).strip()
                if candidate.lower().rstrip('.') not in TITLE_FALSE_POSITIVES and len(candidate) > 5:
                    info['title'] = candidate
        
        # Assignee
        assignee_match = re.search(r"(?:Assignee|Owner)\s*:?\s*([^\n]+)", full_text, re.IGNORECASE)
        if assignee_match:
            assignee = assignee_match.group(1).strip()
            assignee = re.split(r'\s+(?:Date|Filed|Appl)', assignee)[0]
            info['assignee'] = assignee
        
        # Filing date — try standard regex first
        filing_match = re.search(
            r"(?:Filed|Filing\s+Date)\s*:?\s*(\d{1,2}/\d{1,2}/\d{2,4})", full_text, re.IGNORECASE
        )
        if filing_match:
            info['filing_date'] = self._try_parse_date(filing_match.group(1))
        
        # Fix #10: If no filing date found, extract from earliest bookmark title
        # The date of the first bookmark is almost always the filing date
        if not info['filing_date'] and self.bookmarks:
            sorted_bookmarks = sorted(
                [b for b in self.bookmarks if b['page'] is not None],
                key=lambda b: b['page']
            )
            for bm in sorted_bookmarks:
                bm_date = self._extract_date_from_bookmark_title(bm['title'])
                if bm_date:
                    info['filing_date'] = bm_date
                    logger.info(f"Filing date extracted from first bookmark: {bm_date}")
                    break
        
        return info


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        processor = PDFProcessor(sys.argv[1])
        processor.extract_text()
        sections = processor.segment_into_sections()
        
        for section in sections:
            print(f"Document: {section.document_type}")
            print(f"  Pages: {section.page_start + 1} - {section.page_end + 1}")
            print(f"  Date: {section.date}")
            print(f"  High Priority: {section.is_high_priority}")
            print(f"  Text Length: {len(section.text)} chars")
            if section.metadata.get('bookmark_title'):
                print(f"  Bookmark: {section.metadata['bookmark_title']}")
            print()
    else:
        print("Usage: python pdf_processor.py <pdf_path>")
