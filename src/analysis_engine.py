"""
PHAT Analysis Engine - v2.0 (Rebuilt)
Core orchestration for patent prosecution history analysis.

Key improvements:
- Works with bookmark-based document sections from PDFProcessor
- Simplified chunking (only when individual sections exceed AI limits)
- Cleaner error handling and state management
- Integrated OCR cleaning and quote verification
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from .database import (
    DatabaseManager, Patent, Document, Claim, ClaimVersion,
    ProsecutionStatement, RejectionHistory, PriorityBreak,
    TerminalDisclaimer, RestrictionRequirement, AnalysisRun, FinalClaim,
    ValidityRisk, TermSynthesis, ClaimNarrative, PatentTheme,
    TermBoundary, PriorArtReference, ClaimVulnerabilityCard, ProsecutionMilestone,
)
from .ai_providers import AIProvider, AIProviderFactory, ModelSelector, TaskType
from .pdf_processor import PDFProcessor, DocumentSection
from .prompts import (
    SYSTEM_PROMPT, TRIAGE_PROMPT, CLAIMS_EXTRACTION_PROMPT,
    AMENDMENT_ANALYSIS_PROMPT, AMENDMENT_TEXT_RETRY_PROMPT,
    REJECTION_ANALYSIS_PROMPT,
    ARGUMENT_EXTRACTION_PROMPT, RESTRICTION_ANALYSIS_PROMPT,
    ALLOWANCE_ANALYSIS_PROMPT, TERMINAL_DISCLAIMER_PROMPT,
    MEANS_PLUS_FUNCTION_PROMPT, INTERVIEW_SUMMARY_PROMPT,
    COMPREHENSIVE_ANALYSIS_PROMPT,
    SHADOW_EXAMINER_PROMPT, DEFINITION_SYNTHESIS_PROMPT,
    CLAIM_NARRATIVE_PROMPT,
    OFFICE_ACTION_MASTER_PROMPT,
    KEY_LIMITATIONS_PROMPT,
    STRATEGIC_TENSIONS_PROMPT,
    CLAIM_IMPLICATIONS_PROMPT,
    SCOPE_CONSTRAINTS_PROMPT,
    THEMATIC_SYNTHESIS_PROMPT,
    TERM_BOUNDARY_EXTRACTION_PROMPT,
    VULNERABILITY_CARD_PROMPT,
)
from .final_claims import parse_google_patents_claims, compare_claim_texts
from .ocr_cleaner import create_ocr_cleaner
from .verification import ContentVerifier, StatementVerifier, create_content_verifier
from .chunking import DocumentChunker, ResultMerger, create_chunker, ChunkInfo

logger = logging.getLogger(__name__)


def _safe_claim_int(val):
    """Coerce AI-returned claim numbers to int. Returns None if not parseable."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# =============================================================================
# CATEGORY NORMALIZATION (Gap 1 fix)
# Maps LLM-returned relevance category values to canonical categories
# used by the report generator for filtering.
# =============================================================================
CATEGORY_NORMALIZATION = {
    # Prior Art Distinction variants
    'distinction': 'Prior Art Distinction',
    'prior art distinction': 'Prior Art Distinction',
    'art distinction': 'Prior Art Distinction',
    'prior_art_distinction': 'Prior Art Distinction',
    'thematic_distinction': 'Prior Art Distinction',
    'thematic distinction': 'Prior Art Distinction',
    # Purpose/Function characterizations (Fix #6)
    'purpose_characterization': 'Prior Art Distinction',
    'purpose characterization': 'Prior Art Distinction',
    # Conceptual Analogies / Metaphors (Improvement B)
    'conceptual_analogy': 'Prior Art Distinction',
    'conceptual analogy': 'Prior Art Distinction',
    # Definition variants
    'definition': 'Definition/Interpretation',
    'interpretation': 'Definition/Interpretation',
    'definition/interpretation': 'Definition/Interpretation',
    'claim construction': 'Claim Construction',
    # Estoppel variants
    'estoppel': 'Estoppel',
    'scope_limitation': 'Estoppel',
    'scope limitation': 'Estoppel',
    'concession': 'Estoppel',
    'disavowal': 'Estoppel',
    # Traversal
    'traversal': 'Traversal',
    # Reasons for Allowance
    'reasons for allowance': 'Reasons for Allowance',
    'allowance': 'Reasons for Allowance',
    # Interview
    'interview concession': 'Interview Concession',
    'interview agreement': 'Interview Concession',
    # General (fallback)
    'general argument': 'General Argument',
    'general': 'General Argument',
}

# =============================================================================
# GENERIC TERM BLOCKLIST (Gap 7 fix)
# Terms that are too vague to be useful in the claim element lexicon.
# =============================================================================
GENERIC_TERM_BLOCKLIST = {
    'not specified', 'claims generally', 'amended claims', 'invention',
    'the invention', 'general', 'claims', 'invention (general)',
    'claim', 'the claims', 'all claims', 'independent claims',
    'dependent claims', 'the claim', 'n/a', 'none', 'not applicable',
    'the amendment', 'amendment', 'amendments', 'the application',
}

# Regex pattern for claim number references that should not be treated as term definitions
# Matches patterns like "claims 1 and 66", "claim 27", "claims 1-14", etc.
CLAIM_NUMBER_PATTERN = re.compile(r'^claims?\s+\d+(\s*(,|and|&|-|to)\s*\d+)*$', re.IGNORECASE)


# =============================================================================
# PROCEDURAL STATEMENT FILTER (Issue 1 fix)
# Filters out purely procedural statements from estoppel flagging.
# =============================================================================
PROCEDURAL_PHRASES = [
    'would like to discuss', 'sufficiency of the attached',
    'request for interview', 'telephone conversation',
    'please call', 'thank you for your', 'acknowledge receipt',
    'hereby submitted', 'respectfully submitted', 'enclosed herewith',
    'submitting this response', 'filing this response',
    'request for reconsideration', 'entry of this',
    'kindly requested', 'authorized to charge',
    'petition to', 'extension of time',
    'herewith transmits', 'please enter',
]

# =============================================================================
# Gap 9: MILESTONE DOCUMENT TYPES
# These are not analyzed but recorded as prosecution events.
# =============================================================================
MILESTONE_TYPES = {
    "Request for Continued Examination": "RCE",
    "Notice of Appeal": "Appeal",
    "Petition": "Petition",
    "Continuation Filing": "Continuation",
}


# =============================================================================
# Gap 12: CLAIM TYPE CLASSIFIER
# Derives claim type from the preamble text of the issued claim.
# =============================================================================
def classify_claim_type(claim_text: str) -> str:
    """Classify a claim's statutory category from its preamble.
    
    Bug Fix A (v2.2): CRM keywords are checked BEFORE method/process keywords
    because Beauregard claims (computer-readable medium claims) routinely contain
    phrases like "to perform a method" or "embodying a program of instructions...
    to perform a process" in the claim body. Checking method first caused these
    to misclassify as Method claims.
    """
    if not claim_text:
        return 'Unknown'
    preamble = claim_text[:200].lower()
    # Check CRM FIRST — Beauregard claims often contain "method"/"process" in body.
    # Use strong CRM indicators that are unambiguous (not just "memory" which appears
    # in system claims like "a system comprising a processor and a memory").
    strong_crm = [
        'storage medium', 'program storage', 'computer-readable', 'machine-readable',
        'computer readable', 'machine readable', 'non-transitory', 'nontransitory',
        'readable by a computer', 'computer program product',
    ]
    if any(t in preamble for t in strong_crm):
        return 'Computer-Readable Medium'
    # Use word-boundary regex for method/process to avoid matching "processor"
    elif re.search(r'\bmethod\b', preamble) or re.search(r'\bprocess\b', preamble) or 'step of' in preamble:
        return 'Method'
    elif 'composition' in preamble:
        return 'Composition'
    elif 'apparatus' in preamble or 'device' in preamble:
        return 'Apparatus'
    elif 'system' in preamble:
        return 'System'
    # Weak CRM check AFTER system/apparatus — catches edge cases like
    # "A memory storing instructions for..." without a "system" preamble
    elif any(t in preamble for t in ['medium', 'memory', 'storage']):
        return 'Computer-Readable Medium'
    else:
        return 'Unknown'


def is_procedural_statement(text: str) -> bool:
    """Check if a prosecution statement is purely procedural (not substantive).
    
    Issue 1 fix: Procedural statements like interview scheduling and 
    correspondence about sufficiency should not be tagged as estoppel events.
    """
    if not text:
        return True
    text_lower = text.lower().strip()
    # Check for procedural phrases
    if any(phrase in text_lower for phrase in PROCEDURAL_PHRASES):
        # But verify it doesn't ALSO contain substantive content
        substantive_signals = [
            'prior art', 'rejection', 'limitation', 'claim language',
            'does not teach', 'fails to disclose', 'distinguishes',
            'does not render obvious', 'patentable over', 'amended',
        ]
        if not any(sig in text_lower for sig in substantive_signals):
            return True
    # Very short statements are likely procedural
    if len(text_lower) < 30:
        return True
    return False


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ''
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_category(raw_category: str) -> str:
    """Normalize a relevance category string to a canonical value."""
    if not raw_category:
        return 'General Argument'
    key = raw_category.strip().lower()
    return CATEGORY_NORMALIZATION.get(key, raw_category)


def is_generic_term(term: str) -> bool:
    """Check if a claim_element_defined value is too generic to be useful.
    
    Issue 2 fix: Also catches claim number references like 'claims 1 and 66'
    that should not be treated as term definitions.
    """
    if not term:
        return True
    stripped = term.strip()
    if stripped.lower() in GENERIC_TERM_BLOCKLIST or len(stripped) < 3:
        return True
    # Issue 2: Filter claim number references (e.g., "claims 1 and 66")
    if CLAIM_NUMBER_PATTERN.match(stripped):
        return True
    return False


class AnalysisEngine:
    """Main engine for analyzing patent prosecution history"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_manager = DatabaseManager(config['paths']['database'])
        
        # AI providers
        self.model_selector = ModelSelector(config)
        self.ai_provider = AIProviderFactory.create_from_config(config)
        self.fast_ai = AIProviderFactory.create_fast_from_config(config)
        self._task_providers: Dict[TaskType, AIProvider] = {}
        
        # Verbose logging
        self.verbose = config.get('logging', {}).get('verbose', False)
        
        # Chunking (only for sections that exceed AI context)
        self.doc_chunker = create_chunker(config)
        self.result_merger = ResultMerger()
        
        # OCR cleaning
        self.ocr_cleaner = create_ocr_cleaner(config)
        logger.info(f"OCR Cleaner: enabled={self.ocr_cleaner.enabled}")
        
        # Quote verification
        self.verifier = create_content_verifier(config)
        self.statement_verifier = StatementVerifier(self.verifier)
        self.verification_enabled = config.get('verification', {}).get('enabled', True)
        self.downgrade_unverified = config.get('verification', {}).get('downgrade_unverified_risk', True)
        
        # Log model mode
        mode = config.get('model_mode', 'full')
        logger.info(f"Model mode: {mode}")
        if mode == 'custom':
            for task, info in self.model_selector.get_summary().items():
                if task != 'mode':
                    logger.info(f"  {task}: {info}")
        
        # State tracking
        self.current_patent: Optional[Patent] = None
        self.current_claims: Dict[int, Dict] = {}
        self.analysis_run: Optional[AnalysisRun] = None
        
        # Final claims
        self.final_claims_text: Optional[str] = None
        self.final_claims_parsed: Optional[List] = None
        self.final_claims_by_number: Dict[int, Any] = {}
        
        # Interview suggestion tracking
        self.pending_interview_suggestions: List[str] = []
        
        # Verified patent number (from Google Patents or user input)
        self._verified_patent_number: Optional[str] = None
    
    # =========================================================================
    # AI PROVIDER SELECTION
    # =========================================================================
    def _get_provider(self, task: TaskType) -> AIProvider:
        """Get the appropriate AI provider for a task"""
        if self.config.get('model_mode', 'full').lower() != 'custom':
            return self.fast_ai if task == TaskType.TRIAGE else self.ai_provider
        
        if task not in self._task_providers:
            self._task_providers[task] = AIProviderFactory.create_for_task(self.config, task)
        return self._task_providers[task]
    
    # =========================================================================
    # CHUNKED PROCESSING
    # =========================================================================
    def _process_with_chunking(
        self,
        text: str,
        prompt_template: str,
        task_type: TaskType,
        merge_func,
        extra_context: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Process text with automatic chunking if it exceeds AI context limits.
        
        If the assigned model fails (e.g., 400 Bad Request due to context length),
        automatically falls back to the full model before giving up.
        """
        provider = self._get_provider(task_type)
        extra_context = extra_context or {}
        
        if not self.doc_chunker.needs_chunking(text):
            prompt = prompt_template.format(document_text=text, **extra_context)
            try:
                return provider.complete_json(prompt, SYSTEM_PROMPT)
            except Exception as e:
                # Fallback: if the assigned provider fails, try the full model
                if provider is not self.ai_provider:
                    logger.warning(
                        f"Task {task_type.value} failed with assigned model, "
                        f"falling back to full model: {e}"
                    )
                    return self.ai_provider.complete_json(prompt, SYSTEM_PROMPT)
                raise
        
        # Chunk and process
        chunks = self.doc_chunker.chunk_text(text)
        logger.info(f"Chunking {task_type.value}: {len(text)} chars -> {len(chunks)} chunks")
        
        results = []
        for chunk in chunks:
            chunk_note = ""
            if chunk.total_chunks > 1:
                chunk_note = (
                    f"\n[Note: This is part {chunk.chunk_index + 1} of {chunk.total_chunks} "
                    f"of a larger document. Extract all relevant information from this section.]\n"
                )
            
            prompt = prompt_template.format(
                document_text=chunk_note + chunk.text,
                **extra_context,
            )
            
            try:
                result = provider.complete_json(prompt, SYSTEM_PROMPT)
                results.append(result)
            except Exception as e:
                # Fallback per-chunk: try full model before skipping
                if provider is not self.ai_provider:
                    logger.warning(
                        f"Chunk {chunk.chunk_index + 1} failed with assigned model, "
                        f"falling back to full model: {e}"
                    )
                    try:
                        result = self.ai_provider.complete_json(prompt, SYSTEM_PROMPT)
                        results.append(result)
                        continue
                    except Exception as e2:
                        logger.error(f"Chunk {chunk.chunk_index + 1} also failed with full model: {e2}")
                else:
                    logger.error(f"Chunk {chunk.chunk_index + 1} failed: {e}")
        
        if not results:
            raise RuntimeError(f"All chunks failed for {task_type.value}")
        
        return merge_func(results)
    
    # =========================================================================
    # FINAL CLAIMS
    # =========================================================================
    def set_final_claims(self, claims_text: str, parsed_claims: List):
        """Set final claims as ground truth"""
        self.final_claims_text = claims_text
        self.final_claims_parsed = parsed_claims
        self.final_claims_by_number = {c.number: c for c in parsed_claims}
        logger.info(f"Final claims loaded: {len(parsed_claims)} claims")
    
    def set_verified_patent_number(self, patent_number: str):
        """Set a verified patent number (e.g., from Google Patents fetch).
        
        This takes priority over OCR-extracted numbers which may pick up
        provisional application numbers instead of the issued patent number.
        """
        self._verified_patent_number = patent_number
        logger.info(f"Verified patent number set: {patent_number}")
    
    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================
    def analyze_file_history(self, pdf_path: str, resume_patent_id: str = None) -> str:
        """Analyze a patent file history PDF. Returns patent_id.
        
        Args:
            pdf_path: Path to the PDF file
            resume_patent_id: If provided, skip OCR-based patent identification
                              and resume analysis for this known patent ID.
        """
        logger.info(f"Starting analysis: {pdf_path}")
        
        if self.final_claims_parsed:
            logger.info(f"Using {len(self.final_claims_parsed)} final claims as ground truth")
        
        PDFProcessor.set_verbose(self.verbose)
        session = self.db_manager.get_session()
        
        try:
            # Create analysis run
            self.analysis_run = AnalysisRun(
                patent_id="pending",
                ai_provider=self.config['ai_provider'],
                ai_model=self.config['models'][self.config['ai_provider']]['default'],
                start_time=datetime.utcnow(),
                used_final_claims=self.final_claims_parsed is not None,
            )
            session.add(self.analysis_run)
            
            # Process PDF with bookmark-first segmentation
            processor = PDFProcessor(pdf_path, ai_provider=self._get_provider(TaskType.DOCUMENT_CLASSIFICATION))
            processor.extract_text()
            
            # Apply OCR cleaning to extracted text (with garbled text detection + Tesseract fallback)
            if self.ocr_cleaner.enabled:
                logger.info("Applying OCR cleaning to extracted text...")
                
                # Refinement B: Pre-scan bookmarks to identify low-value page ranges
                # (Examiner Search, Citation List) and skip expensive re-OCR on them.
                skip_reocr_pages = set()
                if self.ocr_cleaner.reocr_enabled:
                    processor.extract_bookmarks()
                    if processor.bookmarks:
                        sorted_bm = sorted(
                            [b for b in processor.bookmarks if b.get('page') is not None],
                            key=lambda b: b['page']
                        )
                        LOW_VALUE_KEYWORDS = {'examiner search', 'search strategy', 'citation list', 'search notes'}
                        for i, bm in enumerate(sorted_bm):
                            title_lower = bm.get('title', '').lower()
                            if any(kw in title_lower for kw in LOW_VALUE_KEYWORDS):
                                start_pg = bm['page']
                                end_pg = (sorted_bm[i + 1]['page'] - 1) if i + 1 < len(sorted_bm) else len(processor.text_by_page) - 1
                                for pg in range(start_pg, end_pg + 1):
                                    skip_reocr_pages.add(pg)
                        if skip_reocr_pages:
                            logger.info(f"Refinement B: Skipping re-OCR for {len(skip_reocr_pages)} low-value pages "
                                       f"(Examiner Search / Citation List)")
                
                if self.ocr_cleaner.reocr_enabled:
                    processor.text_by_page = self.ocr_cleaner.clean_pages_with_reocr(
                        processor.text_by_page, pdf_path,
                        skip_reocr_indices=skip_reocr_pages,
                    )
                else:
                    processor.text_by_page = self.ocr_cleaner.clean_pages(processor.text_by_page)
                report = self.ocr_cleaner.get_corrections_report()
                if report['total_corrections'] > 0:
                    logger.info(f"OCR Cleaner made {report['total_corrections']} corrections")
                    self._log_verbose("OCR Correction Report:", report)
                if report.get('reocr_pages_replaced', 0) > 0:
                    logger.info(f"OCR Cleaner re-OCR'd {report['reocr_pages_replaced']} page(s) via Tesseract")
                    self._log_verbose("Re-OCR Details:", report['reocr_details'])
            
            # Segment PDF (bookmarks first, then patterns)
            sections = processor.segment_into_sections()
            logger.info(f"Found {len(sections)} document sections")
            
            # Store source filename for all documents created in this run
            self._source_filename = os.path.basename(pdf_path)
            
            # --- PATENT IDENTIFICATION (with resume support) ---
            if resume_patent_id:
                logger.info(f"Resuming analysis for known Patent ID: {resume_patent_id}")
                self.current_patent = session.query(Patent).get(resume_patent_id)
                if not self.current_patent:
                    raise ValueError(f"Could not find patent with ID {resume_patent_id} to resume")
                
                # Reconstruct claim state from previous run
                self._reconstruct_state(session, self.current_patent)
                
                # Optionally update patent info if we can extract better data now
                patent_info = processor.get_patent_info()
                self._log_verbose("Patent info (resume):", patent_info)
                if patent_info.get('application_number') and not self.current_patent.application_number:
                    self.current_patent.application_number = patent_info['application_number']
                if patent_info.get('patent_number') and not self.current_patent.patent_number:
                    self.current_patent.patent_number = patent_info['patent_number']
                if patent_info.get('title') and not self.current_patent.title:
                    self.current_patent.title = patent_info['title']
                session.flush()
            else:
                # Original logic: extract patent info and find/create record
                patent_info = processor.get_patent_info()
                self._log_verbose("Patent info:", patent_info)
                
                # Override with verified patent number if available
                # (e.g., from Google Patents fetch — more reliable than OCR)
                if self._verified_patent_number:
                    patent_info['patent_number'] = self._verified_patent_number
                    logger.info(f"Using verified patent number: {self._verified_patent_number}")
                
                self.current_patent = self._find_or_create_patent(session, patent_info)
            
            # Store final claims if provided
            if self.final_claims_text and self.final_claims_parsed:
                self._store_final_claims(session)
            
            self.analysis_run.patent_id = self.current_patent.id
            
            # Sort sections chronologically
            dated_sections = sorted(sections, key=lambda s: s.date or datetime.min)
            
            # Filter relevant sections
            skip_types = self.config.get('triage', {}).get('skip_types', [])
            # Types that should NEVER be processed regardless of content
            # (e.g. fee worksheets that mention "35 U.S.C." in boilerplate text)
            HARD_SKIP = {
                "Fee Transmittal", "Power of Attorney", "Application Data Sheet",
                "Filing Receipt", "Administrative", "Drawings", "Transmittal",
                "Examiner Search", "Citation List",
            }
            relevant = []
            skipped = 0
            for s in dated_sections:
                if s.document_type in HARD_SKIP:
                    skipped += 1
                elif s.document_type in skip_types:
                    # For other skip types (e.g. IDS), check for substantive content
                    if self._contains_substantive_content(s.text):
                        logger.warning(f"'{s.document_type}' has substantive content - processing anyway")
                        relevant.append(s)
                    else:
                        skipped += 1
                else:
                    relevant.append(s)
            
            if skipped:
                logger.info(f"Skipped {skipped} boilerplate sections")
            logger.info(f"Processing {len(relevant)} relevant sections")
            
            # Process each section
            for i, section in enumerate(relevant):
                if self._is_already_processed(session, section):
                    logger.info(f"[{i+1}/{len(relevant)}] Skipping (already processed): {section.document_type}")
                    continue
                
                logger.info(f"[{i+1}/{len(relevant)}] Processing: {section.document_type} "
                           f"(pages {section.page_start+1}-{section.page_end+1})")
                self._process_section(session, section)
                self.analysis_run.documents_processed = i + 1
                session.commit()
            
            # Post-processing: map prosecution claims to final claims
            if self.final_claims_parsed:
                self._map_claims_to_final(session)
            
            # Verification summary
            if self.verification_enabled:
                ver_log = self.statement_verifier.get_verification_log()
                if ver_log:
                    verified = sum(1 for v in ver_log if v['status'] == 'verified')
                    unverified = sum(1 for v in ver_log if v['status'] == 'unverified')
                    logger.info(f"Verification: {verified} verified, {unverified} unverified / {len(ver_log)} total")
            
            # =================================================================
            # POST-PROCESSING: Layer 2/3 Synthesis & Critique
            # Runs AFTER all raw extraction is committed.
            # =================================================================
            post_process_enabled = self.config.get('post_processing', {}).get('enabled', True)
            if post_process_enabled:
                logger.info("Starting post-processing (synthesis & critique)...")
                try:
                    self._post_process_data(session)
                    self.analysis_run.post_processing_completed = True
                    session.commit()
                    logger.info("Post-processing completed successfully")
                except Exception as e:
                    logger.error(f"Post-processing failed (raw data is intact): {e}", exc_info=True)
                    # Don't fail the entire run — raw data is already committed
                    self.analysis_run.post_processing_completed = False
            
            # Finalize
            self.analysis_run.end_time = datetime.utcnow()
            self.analysis_run.status = "completed"
            session.commit()
            
            logger.info("Analysis completed successfully")
            return self.current_patent.id
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            if self.analysis_run:
                self.analysis_run.status = "failed"
                self.analysis_run.errors = {"error": str(e)}
            session.rollback()
            raise
        finally:
            session.close()
    
    # =========================================================================
    # PATENT RECORD MANAGEMENT
    # =========================================================================
    def _find_or_create_patent(self, session, patent_info: Dict) -> Patent:
        """Find existing patent record or create a new one"""
        candidates = []
        
        if patent_info.get('application_number'):
            found = session.query(Patent).filter(
                Patent.application_number.like(f"{patent_info['application_number']}%")
            ).all()
            candidates.extend(found)
        
        if patent_info.get('patent_number'):
            found = session.query(Patent).filter(
                Patent.patent_number.like(f"{patent_info['patent_number']}%")
            ).all()
            candidates.extend(found)
        
        # Deduplicate
        unique = {p.id: p for p in candidates}.values()
        
        if unique:
            # Pick the one with most progress
            existing = max(unique, key=lambda p: sum(1 for d in p.documents if d.is_processed))
            logger.info(f"Resuming existing patent: {existing.application_number or existing.patent_number}")
            
            # Update with better info if available
            if patent_info.get('application_number') and not existing.application_number:
                existing.application_number = patent_info['application_number']
            if patent_info.get('patent_number') and not existing.patent_number:
                existing.patent_number = patent_info['patent_number']
            
            session.flush()
            self._reconstruct_state(session, existing)
            return existing
        
        # Create new
        patent = Patent(
            patent_number=patent_info.get('patent_number'),
            application_number=patent_info.get('application_number'),
            title=patent_info.get('title'),
            filing_date=patent_info.get('filing_date'),
        )
        session.add(patent)
        session.flush()
        logger.info(f"Created new patent record: {patent.id}")
        return patent
    
    def _reconstruct_state(self, session, patent: Patent):
        """Reconstruct claim state from database for resuming analysis"""
        self.current_claims = {}
        claims = session.query(Claim).filter(Claim.patent_id == patent.id).all()
        
        for claim in claims:
            if claim.current_status == 'Cancelled':
                continue
            latest = session.query(ClaimVersion).filter(
                ClaimVersion.claim_id == claim.id
            ).order_by(ClaimVersion.version_number.desc()).first()
            
            if latest:
                self.current_claims[claim.application_claim_number] = {
                    'id': claim.id,
                    'text': latest.claim_text,
                    'version': latest.version_number,
                }
        
        logger.info(f"Reconstructed state: {len(self.current_claims)} active claims")
    
    # =========================================================================
    # FINAL CLAIMS STORAGE & MAPPING
    # =========================================================================
    def _store_final_claims(self, session):
        """Store final claims in the database"""
        self.current_patent.final_claims_text = self.final_claims_text
        self.current_patent.final_claims_json = [
            {'number': c.number, 'text': c.text, 'is_independent': c.is_independent, 'depends_on': c.depends_on}
            for c in self.final_claims_parsed
        ]
        self.current_patent.has_final_claims = True
        
        # If we have a verified patent number and the current one is missing
        # or looks like a provisional (e.g., "60/816,844"), override it
        if self._verified_patent_number:
            current = self.current_patent.patent_number or ''
            is_provisional = bool(re.match(r'^\d{2}/', current))
            if not current or is_provisional:
                logger.info(f"Overriding patent number '{current}' with verified '{self._verified_patent_number}'")
                self.current_patent.patent_number = self._verified_patent_number
        
        session.query(FinalClaim).filter(FinalClaim.patent_id == self.current_patent.id).delete()
        
        for claim in self.final_claims_parsed:
            fc = FinalClaim(
                patent_id=self.current_patent.id,
                claim_number=claim.number,
                claim_text=claim.text,
                normalized_text=normalize_text(claim.text),
                is_independent=claim.is_independent,
                depends_on=claim.depends_on,
            )
            session.add(fc)
        
        session.flush()
        logger.info(f"Stored {len(self.final_claims_parsed)} final claims")
    
    def _map_claims_to_final(self, session):
        """Map prosecution claims to final issued claims"""
        claims = session.query(Claim).filter(Claim.patent_id == self.current_patent.id).all()
        final_claims = session.query(FinalClaim).filter(FinalClaim.patent_id == self.current_patent.id).all()
        
        if not claims or not final_claims:
            return
        
        final_by_num = {fc.claim_number: fc for fc in final_claims}
        mapped_count = 0
        
        # =====================================================================
        # Fix #13: Trust existing final_issued_number mappings
        # If allowance processing already determined the issued claim number
        # (via renumbering table or explicit mapping), trust that at 100%
        # confidence instead of re-computing via text similarity.
        # =====================================================================
        for claim in claims:
            if claim.final_issued_number and claim.final_issued_number in final_by_num:
                fc = final_by_num[claim.final_issued_number]
                if not fc.mapped_app_claim_id:
                    claim.mapped_final_claim_number = claim.final_issued_number
                    claim.mapping_confidence = 1.0  # Trust the explicit mapping
                    fc.mapped_app_claim_id = claim.id
                    fc.mapping_confidence = 1.0
                    mapped_count += 1
                    logger.info(f"Trusted existing mapping: App Claim {claim.application_claim_number} "
                               f"-> Issued Claim {claim.final_issued_number} (100% confidence)")
                    continue
        
        for claim in claims:
            # Fix #13: Skip claims already mapped by trusted explicit mappings above
            if claim.mapped_final_claim_number and claim.mapping_confidence == 1.0:
                continue
            
            latest = session.query(ClaimVersion).filter(
                ClaimVersion.claim_id == claim.id
            ).order_by(ClaimVersion.version_number.desc()).first()
            
            if not latest:
                continue
            
            app_num = claim.application_claim_number
            
            # Try exact number match first
            if app_num in final_by_num:
                fc = final_by_num[app_num]
                comparison = compare_claim_texts(latest.claim_text, fc.claim_text)
                if comparison['similarity_score'] > 0.5:
                    claim.mapped_final_claim_number = app_num
                    claim.mapping_confidence = comparison['similarity_score']
                    claim.final_issued_number = app_num
                    fc.mapped_app_claim_id = claim.id
                    fc.mapping_confidence = comparison['similarity_score']
                    mapped_count += 1
                    continue
            
            # Try best text match
            best_match = None
            best_score = 0.5
            for fc in final_claims:
                if fc.mapped_app_claim_id:
                    continue
                comparison = compare_claim_texts(latest.claim_text, fc.claim_text)
                if comparison['similarity_score'] > best_score:
                    best_score = comparison['similarity_score']
                    best_match = fc
            
            if best_match:
                claim.mapped_final_claim_number = best_match.claim_number
                claim.mapping_confidence = best_score
                claim.final_issued_number = best_match.claim_number
                best_match.mapped_app_claim_id = claim.id
                best_match.mapping_confidence = best_score
                mapped_count += 1
        
        # =====================================================================
        # FALLBACK: Ordered mapping for unmapped claims (Gap 3 fix)
        # If text similarity failed but we have allowed claims and final claims
        # of the same count, map them by sequential order — this is how patent
        # offices actually renumber claims.
        # =====================================================================
        unmapped_final = [fc for fc in final_claims if not fc.mapped_app_claim_id]
        if unmapped_final:
            # Get allowed claims sorted by application claim number
            allowed_unmapped = sorted(
                [c for c in claims
                 if c.current_status == 'Allowed'
                 and not c.mapped_final_claim_number],
                key=lambda c: c.application_claim_number
            )
            
            # Sort unmapped final claims by claim number
            unmapped_final_sorted = sorted(unmapped_final, key=lambda fc: fc.claim_number)
            
            if len(allowed_unmapped) >= len(unmapped_final_sorted):
                logger.info(f"Fallback mapping: {len(unmapped_final_sorted)} unmapped final claims "
                           f"to {len(allowed_unmapped)} allowed application claims by order")
                for fc, claim in zip(unmapped_final_sorted, allowed_unmapped):
                    claim.mapped_final_claim_number = fc.claim_number
                    claim.mapping_confidence = 0.4  # Lower confidence for order-based mapping
                    claim.final_issued_number = fc.claim_number
                    fc.mapped_app_claim_id = claim.id
                    fc.mapping_confidence = 0.4
                    mapped_count += 1
                    logger.info(f"  App Claim {claim.application_claim_number} -> "
                               f"Issued Claim {fc.claim_number} (order-based)")
        
        logger.info(f"Claim mapping: {mapped_count} claims mapped to final issued claims")
        session.flush()
    
    # =========================================================================
    # SECTION PROCESSING
    # =========================================================================
    def _is_already_processed(self, session, section: DocumentSection) -> bool:
        return session.query(Document).filter(
            Document.patent_id == self.current_patent.id,
            Document.page_start == section.page_start,
            Document.page_end == section.page_end,
            Document.document_type == section.document_type,
            Document.is_processed == True,
        ).count() > 0
    
    def _process_section(self, session, section: DocumentSection):
        """Process a single document section"""
        # Create document record
        doc = Document(
            patent_id=self.current_patent.id,
            document_type=section.document_type,
            document_date=section.date,
            page_start=section.page_start,
            page_end=section.page_end,
            is_high_priority=section.is_high_priority,
            raw_text=section.text[:50000],
            filename=getattr(self, '_source_filename', None),
        )
        session.add(doc)
        session.flush()
        
        # Triage: quick relevance check
        if not self._should_process(section):
            # Gap 9: Still record milestone events even though we skip full analysis
            if section.document_type in MILESTONE_TYPES:
                milestone = ProsecutionMilestone(
                    patent_id=self.current_patent.id,
                    milestone_type=MILESTONE_TYPES[section.document_type],
                    date=section.date,
                    context=f"{section.document_type} filed",
                    document_id=doc.id,
                )
                session.add(milestone)
                logger.info(f"Recorded milestone: {MILESTONE_TYPES[section.document_type]} ({section.date})")
            doc.is_processed = True
            return
        
        # Route to handler
        handlers = {
            "Office Action": self._process_office_action,
            "Amendment": self._process_amendment,
            "Appeal Brief": self._process_appeal_brief,
            "Notice of Allowance": self._process_allowance,
            "Restriction Requirement": self._process_restriction,
            "Interview Summary": self._process_interview,
            "Claims": self._process_initial_claims,
            "Terminal Disclaimer": self._process_terminal_disclaimer,
            "PTAB Decision": self._process_ptab_decision,
            "Reasons for Allowance": self._process_reasons_for_allowance,
        }
        
        handler = handlers.get(section.document_type, self._process_generic)
        
        try:
            handler(session, doc, section)
        except Exception as e:
            logger.error(f"Handler failed for {section.document_type}: {e}", exc_info=True)
        
        doc.is_processed = True
    
    def _should_process(self, section: DocumentSection) -> bool:
        """
        Quick triage check.
        OPTIMIZATION: Trust regex/bookmark classification for obvious exclusions
        to save API calls. Only use AI triage for substantive or unknown types.
        """
        # 1. Hard-coded skip list (trust the PDFProcessor's classification)
        ALWAYS_SKIP = [
            "Fee Transmittal", "Power of Attorney", "Application Data Sheet",
            "Filing Receipt", "Information Disclosure Statement",
            "Index of Claims", "Search Report", "Transmittal Letter",
            "Administrative", "Drawings", "Transmittal",
            "Examiner Search", "Citation List", "Correspondence",
            # Gap 9: "Request for Continued Examination" removed — now tracked as milestone
        ]
        
        if section.document_type in ALWAYS_SKIP:
            logger.info(f"Skipping '{section.document_type}' based on classification (no API call)")
            return False
        
        # 1b. Gap 9: Milestone types — record but don't fully analyze
        if section.document_type in MILESTONE_TYPES:
            logger.info(f"Recording milestone: '{section.document_type}' (no full analysis)")
            # Return False to skip full analysis; milestone recorded in _process_section
            return False
        
        # 2. Known substantive types — bypass triage (trust PDFProcessor classification)
        KNOWN_SUBSTANTIVE = [
            "Office Action", "Amendment", "Appeal Brief", "Reply Brief",
            "Notice of Allowance", "Statement of Reasons for Allowance",
            "Restriction Requirement", "Interview Summary",
            "PTAB Decision", "Ex Parte Quayle Action",
            "Claims", "Terminal Disclaimer", "Reasons for Allowance",
            "Specification",
        ]
        
        if section.document_type in KNOWN_SUBSTANTIVE:
            logger.info(f"Skipping triage for known substantive type: '{section.document_type}'")
            return True
        
        # 3. Heuristic: too short to be substantive?
        if len(section.text.strip()) < 200:
            logger.info(f"Skipping section (text too short: {len(section.text.strip())} chars)")
            return False
        
        # 4. Only use AI triage for substantive or unknown document types
        try:
            prompt = TRIAGE_PROMPT.format(document_text=section.text[:5000])
            result = self._get_provider(TaskType.TRIAGE).complete_json(prompt, SYSTEM_PROMPT)
            return result.get('is_relevant', True)
        except Exception as e:
            logger.warning(f"Triage failed, defaulting to process: {e}")
            return True
    
    def _contains_substantive_content(self, text: str) -> bool:
        """Check if text contains substantive patent prosecution content"""
        patterns = [
            r"claim\s+\d+\s+is\s+rejected",
            r"what\s+is\s+claimed\s+is",
            r"in\s+response\s+to\s+the\s+office\s+action",
            r"applicant.?s?\s+argument",
            r"under\s+35\s+u\.?s\.?c",
            r"restriction\s+requirement",
            r"notice\s+of\s+allowance",
            r"terminal\s+disclaimer",
            r"interview\s+summary",
            r"appeal\s+brief",
        ]
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)
    
    # =========================================================================
    # DOCUMENT TYPE HANDLERS
    # =========================================================================
    
    def _process_initial_claims(self, session, doc: Document, section: DocumentSection):
        """Process initial claims from application"""
        logger.info("Processing initial claims")
        
        result = self._process_with_chunking(
            text=section.text,
            prompt_template=CLAIMS_EXTRACTION_PROMPT,
            task_type=TaskType.CLAIMS_EXTRACTION,
            merge_func=self.result_merger.merge_claims,
        )
        
        for claim_data in result.get('claims', []):
            claim_num = _safe_claim_int(claim_data.get('number'))
            if not claim_num or claim_num in self.current_claims:
                continue
            
            claim = Claim(
                patent_id=self.current_patent.id,
                application_claim_number=claim_num,
                is_independent=claim_data.get('is_independent', True),
                current_status='Pending',
            )
            
            if claim_data.get('depends_on'):
                parent = session.query(Claim).filter(
                    Claim.patent_id == self.current_patent.id,
                    Claim.application_claim_number == _safe_claim_int(claim_data['depends_on']),
                ).first()
                if parent:
                    claim.parent_claim_id = parent.id
                    claim.family_tree_id = parent.family_tree_id or parent.id
            
            session.add(claim)
            session.flush()
            
            if claim.is_independent:
                claim.family_tree_id = claim.id
            
            version = ClaimVersion(
                claim_id=claim.id,
                document_id=doc.id,
                version_number=1,
                claim_text=claim_data.get('text') or '',
                normalized_text=normalize_text(claim_data.get('text') or ''),
                date_of_change=section.date,
            )
            session.add(version)
            
            self.current_claims[claim_num] = {
                'id': claim.id,
                'text': claim_data.get('text') or '',
                'version': 1,
            }
        
        logger.info(f"Extracted {len(self.current_claims)} claims")
    
    def _process_office_action(self, session, doc: Document, section: DocumentSection):
        """Process an Office Action using consolidated single-call prompt"""
        logger.info("Processing Office Action (consolidated prompt)")
        
        # Single LLM call replaces 3 separate calls
        result = self._process_with_chunking(
            text=section.text,
            prompt_template=OFFICE_ACTION_MASTER_PROMPT,
            task_type=TaskType.REJECTION_ANALYSIS,
            merge_func=self.result_merger.merge_rejections,
        )
        
        # --- Process rejections (same logic as before) ---
        
        # Update document date if the LLM extracted one (Gap 6 fix)
        if result.get('document_date') and not doc.document_date:
            try:
                doc.document_date = datetime.strptime(result['document_date'], "%Y-%m-%d")
            except (ValueError, TypeError):
                pass
        
        for rej_data in result.get('rejections', []):
            rationale = rej_data.get('examiner_rationale')
            if rationale is not None and not isinstance(rationale, str):
                rationale = json.dumps(rationale) if isinstance(rationale, (list, dict)) else str(rationale)
            
            # Include motivation_to_combine in rationale for 103 rejections
            motivation = rej_data.get('motivation_to_combine')
            if motivation and rej_data.get('is_103_combination'):
                rationale = f"{rationale or ''} [Motivation to combine: {motivation}]".strip()
            
            rejection = RejectionHistory(
                document_id=doc.id,
                claim_number=rej_data['affected_claims'][0] if rej_data.get('affected_claims') else 0,
                rejection_type=rej_data.get('type', 'rejection'),
                statutory_basis=rej_data.get('statutory_basis'),
                cited_prior_art=rej_data.get('prior_art'),
                rejected_claim_elements=rej_data.get('rejected_elements'),
                rejection_rationale=rationale,
                is_final=result.get('is_final_action', False),
            )
            session.add(rejection)
            
            for claim_num in rej_data.get('affected_claims', []):
                if claim_num in self.current_claims:
                    claim = session.query(Claim).get(self.current_claims[claim_num]['id'])
                    if claim:
                        claim.current_status = 'Rejected'
        
        # --- Process objections (same logic as before) ---
        for obj_data in result.get('objections', []):
            for claim_num in obj_data.get('affected_claims', []):
                if claim_num in self.current_claims:
                    claim = session.query(Claim).get(self.current_claims[claim_num]['id'])
                    if claim and claim.current_status == 'Pending':
                        claim.current_status = 'Objected To'
        
        # --- Process examiner definitions as ProsecutionStatement rows ---
        for defn in result.get('examiner_definitions', []):
            stmt = ProsecutionStatement(
                document_id=doc.id,
                speaker='Examiner',
                extracted_text=f"{defn.get('term', '')}: {defn.get('interpretation', '')}",
                relevance_category='Examiner Definition',
                claim_element_defined=defn.get('term'),
                affected_claims=defn.get('affected_claims'),
                is_acquiesced=True,
                context_summary=f"Examiner interprets '{defn.get('term', '')}' as: {defn.get('interpretation', '')}",
            )
            session.add(stmt)
        
        # --- Process means-plus-function issues (same logic as _check_means_plus_function) ---
        for mpf in result.get('means_plus_function_issues', []):
            claim_num = mpf.get('claim_number')
            if claim_num in self.current_claims:
                claim = session.query(Claim).get(self.current_claims[claim_num]['id'])
                if claim:
                    claim.is_means_plus_function = True
                    existing = claim.means_plus_function_elements or []
                    existing.append(mpf)
                    claim.means_plus_function_elements = existing
        
        # --- Process allowable claims ---
        for claim_num in result.get('allowable_claims', []):
            if claim_num in self.current_claims:
                claim = session.query(Claim).get(self.current_claims[claim_num]['id'])
                if claim:
                    claim.current_status = 'Indicated Allowable'
        
        # NOTE: _extract_arguments() and _check_means_plus_function() are NOT called
        # separately — the consolidated prompt handles all three tasks in one API call.
    
    def _process_amendment(self, session, doc: Document, section: DocumentSection):
        """Process an Amendment (MOST CRITICAL handler)"""
        logger.info("Processing Amendment")
        
        # Interview handshake: correlate with pending suggestions
        # Fix #7: More robust matching — also check for partial matches and
        # store the link between suggestion and the specific amendment
        if self.pending_interview_suggestions:
            section_text_lower = section.text.lower()
            for suggestion in self.pending_interview_suggestions:
                if not suggestion or len(suggestion) <= 10:
                    continue
                suggestion_lower = suggestion.lower()
                # Check for exact substring match
                exact_match = suggestion_lower in section_text_lower
                # Also check for key phrase overlap (words from suggestion in text)
                suggestion_words = set(w for w in suggestion_lower.split() if len(w) > 4)
                word_overlap = sum(1 for w in suggestion_words if w in section_text_lower)
                partial_match = len(suggestion_words) > 0 and word_overlap / len(suggestion_words) >= 0.6
                
                if exact_match or partial_match:
                    match_type = "exact" if exact_match else "partial"
                    stmt = ProsecutionStatement(
                        document_id=doc.id,
                        speaker='Applicant',
                        extracted_text=f"Amendment incorporates examiner suggestion: '{suggestion}'",
                        relevance_category='Examiner-Defined Scope',
                        is_acquiesced=True,
                        context_summary=f"Adopted examiner suggestion from previous interview ({match_type} match). "
                                       f"The examiner suggested this amendment, creating strong estoppel.",
                    )
                    session.add(stmt)
                    logger.info(f"Linked interview suggestion to amendment ({match_type}): '{suggestion[:60]}...'")
            self.pending_interview_suggestions = []
        
        # Build context
        prev_claims_text = json.dumps(self.current_claims, indent=2)
        prior_art_list = self._get_prior_art_context(session, doc)
        
        # Get active rejections for linking
        active_rejections = self._get_active_rejections(session, doc)
        
        # Analyze amendments
        result = self._process_with_chunking(
            text=section.text,
            prompt_template=AMENDMENT_ANALYSIS_PROMPT,
            task_type=TaskType.AMENDMENT_ANALYSIS,
            merge_func=self.result_merger.merge_amendments,
            extra_context={
                'previous_claims': prev_claims_text[:5000],
                'prior_art_list': prior_art_list,
            },
        )
        
        # =====================================================================
        # Validate: check for amended claims missing current_text.
        # If found, fire a targeted retry prompt to recover the full text.
        # =====================================================================
        missing_text_claims = []
        for amend in result.get('amended_claims', []):
            if amend.get('change_type') == 'amended' and not (amend.get('current_text') or '').strip():
                claim_num = amend.get('claim_number')
                if claim_num and claim_num in self.current_claims:
                    missing_text_claims.append(amend)
        
        if missing_text_claims:
            logger.warning(
                f"Amendment analysis missing current_text for {len(missing_text_claims)} claims: "
                f"{[a['claim_number'] for a in missing_text_claims]}. Attempting targeted retry..."
            )
            try:
                # Build context for the retry prompt
                missing_info = []
                prev_texts = {}
                for amend in missing_text_claims:
                    cn = amend['claim_number']
                    prev_text = self.current_claims.get(cn, {}).get('text', '')
                    summary = amend.get('change_summary', 'unknown changes')
                    missing_info.append(f"- Claim {cn}: {summary}")
                    prev_texts[cn] = prev_text
                
                prev_claims_for_retry = "\n\n".join(
                    f"Claim {cn} (before amendment):\n{text}"
                    for cn, text in prev_texts.items()
                )
                
                retry_prompt = AMENDMENT_TEXT_RETRY_PROMPT.format(
                    document_text=section.text[:50000],
                    previous_claims_text=prev_claims_for_retry[:10000],
                    missing_claims="\n".join(missing_info),
                )
                
                provider = self._get_provider(TaskType.AMENDMENT_ANALYSIS)
                retry_result = provider.complete_json(retry_prompt, SYSTEM_PROMPT)
                
                # Patch the missing texts back into the original result
                recovered = {}
                for claim_data in retry_result.get('claims', []):
                    cn = claim_data.get('claim_number')
                    text = claim_data.get('current_text', '').strip()
                    if cn and text:
                        recovered[cn] = text
                
                for amend in result['amended_claims']:
                    cn = amend.get('claim_number')
                    if cn in recovered and not (amend.get('current_text') or '').strip():
                        amend['current_text'] = recovered[cn]
                        logger.info(f"Claim {cn}: Recovered full text via retry ({len(recovered[cn])} chars)")
                
                still_missing = [
                    a['claim_number'] for a in result['amended_claims']
                    if a.get('change_type') == 'amended'
                    and not (a.get('current_text') or '').strip()
                    and a.get('claim_number') in {m['claim_number'] for m in missing_text_claims}
                ]
                if still_missing:
                    logger.warning(f"Retry could not recover text for claims: {still_missing}")
                else:
                    logger.info(f"Successfully recovered all {len(recovered)} missing claim texts")
                    
            except Exception as e:
                logger.error(f"Targeted retry for missing claim texts failed: {e}")
        
        # Pre-fetch cancelled claim texts for zombie analysis
        cancelled_texts = {}
        for amend in result.get('amended_claims', []):
            if amend.get('change_type') == 'cancelled':
                c_num = amend.get('claim_number')
                if c_num in self.current_claims:
                    cancelled_texts[c_num] = self.current_claims[c_num]['text']
        
        # Process each amendment
        for amend in result.get('amended_claims', []):
            claim_num = amend.get('claim_number')
            change_type = amend.get('change_type', '')
            
            if change_type == 'cancelled':
                self._handle_cancel(session, claim_num)
            elif change_type == 'new':
                self._handle_new_claim(session, doc, section, amend, active_rejections)
            elif change_type == 'amended' and claim_num in self.current_claims:
                self._handle_amendment(session, doc, section, amend, cancelled_texts, active_rejections)
        
        # Handle renumbering
        for renum in result.get('renumbering', []):
            old, new = renum.get('old_number'), renum.get('new_number')
            if old in self.current_claims and old != new:
                self.current_claims[new] = self.current_claims.pop(old)
        
        # NOTE: We no longer call _extract_arguments here — the combined prompt
        # captures both amendment details and arguments in a single API call.
        
        # Update document date if the LLM extracted one (Gap 6 fix)
        if result.get('document_date') and not doc.document_date:
            try:
                doc.document_date = datetime.strptime(result['document_date'], "%Y-%m-%d")
            except (ValueError, TypeError):
                pass
        
        # Process general_arguments from the combined prompt
        # This replaces the separate _extract_arguments call for Amendment docs,
        # saving an API call while preserving the causal link between amendments and arguments.
        for arg in result.get('general_arguments', []):
            extracted_text = arg.get('text', '')
            
            # Issue 1 fix: Filter purely procedural statements
            if is_procedural_statement(extracted_text):
                logger.info(f"Filtering procedural statement: '{extracted_text[:60]}...'")
                continue
            
            # Quote verification on general arguments
            if self.verification_enabled and extracted_text:
                ver = self.verifier.verify_quote(section.text, extracted_text)
                
                self.statement_verifier.verification_log.append({
                    'quote': extracted_text[:100] + '...' if len(extracted_text) > 100 else extracted_text,
                    'status': ver.status.value,
                    'confidence': ver.confidence,
                })
                
                if ver.is_suspicious:
                    logger.warning(f"Potential hallucination in general argument: '{extracted_text[:50]}...'")
                    existing_ctx = arg.get('context', '') or ''
                    arg['context'] = f"[WARNING: Quote not verified] {existing_ctx}".strip()
                    if self.downgrade_unverified:
                        arg['estoppel_risk'] = 'unknown'
                
                if ver.matched_text and ver.is_verified:
                    extracted_text = ver.matched_text
            
            # Disavowal confidence filtering for general arguments
            if arg.get('negative_limitation') and arg.get('disavowal_confidence') == 'low':
                logger.info(f"Filtering low-confidence disavowal in general argument: '{extracted_text[:50]}...'")
                continue  # Skip this statement entirely
            
            context = arg.get('context', '')
            if arg.get('negative_limitation') and arg.get('disavowal_confidence') == 'medium':
                context = f"[UNCONFIRMED DISAVOWAL] {context}".strip()
            
            # Fix #4: Flag directionality constraints prominently
            if arg.get('directionality_constraint'):
                context = f"[DIRECTIONALITY CONSTRAINT] {context}".strip()
            
            # Fix #6: Flag purpose/function characterizations
            if arg.get('statement_type') == 'purpose_characterization':
                context = f"[PURPOSE/FUNCTION CHARACTERIZATION] {context}".strip()
            
            # Normalize the relevance category (Gap 1 fix)
            raw_category = arg.get('statement_type', 'General Argument')
            normalized_category = normalize_category(raw_category)
            
            # Filter generic claim element terms (Gap 7 fix)
            claim_element = arg.get('claim_element_defined') or arg.get('claim_element')
            if is_generic_term(claim_element):
                claim_element = None
            
            # Build cited_prior_art list
            cited_art = None
            if arg.get('cited_prior_art'):
                art = arg['cited_prior_art']
                if isinstance(art, list):
                    cited_art = art
                elif isinstance(art, str) and art.lower() not in ('none', 'null', 'n/a', ''):
                    cited_art = [art]
            
            stmt = ProsecutionStatement(
                document_id=doc.id,
                speaker=arg.get('speaker', 'Applicant'),
                extracted_text=extracted_text,
                relevance_category=normalized_category,
                claim_element_defined=claim_element,
                context_summary=context,
                cited_prior_art=cited_art,
                affected_claims=arg.get('affected_claims'),
                traversal_present=arg.get('traversal_present', False),
                is_acquiesced=arg.get('speaker') == 'Examiner',
            )
            session.add(stmt)
    
    def _handle_cancel(self, session, claim_num):
        """Cancel a claim"""
        if claim_num in self.current_claims:
            claim = session.query(Claim).get(self.current_claims[claim_num]['id'])
            if claim:
                claim.current_status = 'Cancelled'
            del self.current_claims[claim_num]
    
    def _handle_new_claim(self, session, doc, section, amend, active_rejections):
        """Handle a new claim added by amendment"""
        claim_num = _safe_claim_int(amend['claim_number'])
        claim_text = amend.get('current_text') or ''
        
        claim = Claim(
            patent_id=self.current_patent.id,
            application_claim_number=claim_num,
            is_independent='independent' in claim_text.lower()[:100] if claim_text else True,
            current_status='Pending',
        )
        session.add(claim)
        session.flush()
        
        version = ClaimVersion(
            claim_id=claim.id,
            document_id=doc.id,
            version_number=1,
            claim_text=claim_text,
            normalized_text=normalize_text(claim_text),
            change_summary='New claim',
            date_of_change=section.date,
            added_limitations=amend.get('added_limitations'),
        )
        session.add(version)
        
        # Link to rejections
        self._link_limitations_to_rejections(amend, active_rejections, version)
        
        self.current_claims[claim_num] = {
            'id': claim.id,
            'text': claim_text,
            'version': 1,
        }
    
    def _handle_amendment(self, session, doc, section, amend, cancelled_texts, active_rejections):
        """Handle an amended claim"""
        claim_num = _safe_claim_int(amend['claim_number'])
        claim_data = self.current_claims[claim_num]
        claim = session.query(Claim).get(claim_data['id'])
        
        new_text = amend.get('current_text') or ''
        change_summary = amend.get('change_summary', '') or ''
        
        # Fix #9 (refined): Handle missing current_text based on change_summary quality
        if not new_text.strip() and claim_data.get('text'):
            # Detect likely false positives — the model flagged an amendment it can't substantiate
            FALSE_POSITIVE_SIGNALS = [
                'not specified', 'no change', 'no amendment', 'unchanged',
                'not clear', 'cannot determine', 'unable to determine',
            ]
            is_likely_false_positive = (
                not change_summary.strip()
                or any(sig in change_summary.lower() for sig in FALSE_POSITIVE_SIGNALS)
            )
            
            if is_likely_false_positive:
                # Skip this entirely — don't create a phantom version
                logger.warning(
                    f"Claim {claim_num}: Skipping likely false-positive amendment "
                    f"(no current_text, change_summary: '{change_summary[:80]}')"
                )
                return
            else:
                # Legitimate amendment but model omitted full text — use previous text
                # and mark the version so the report can flag it
                new_text = claim_data['text']
                logger.info(
                    f"Claim {claim_num}: Model omitted current_text, using previous "
                    f"version text. Change: {change_summary[:80]}"
                )
                # Annotate the change summary so the report shows this is reconstructed
                if '[text not provided by model]' not in change_summary:
                    change_summary = f"{change_summary} [text not provided by model — showing previous version]"
                amend['change_summary'] = change_summary
        
        old_normalized = normalize_text(claim_data['text'])
        new_normalized = normalize_text(new_text)
        is_substantive = old_normalized != new_normalized
        
        # Zombie claim check
        if is_substantive and claim and claim.is_independent and cancelled_texts:
            for c_num, c_text in cancelled_texts.items():
                similarity = compare_claim_texts(c_text, new_text)
                if similarity['similarity_score'] > 0.8:
                    stmt = ProsecutionStatement(
                        document_id=doc.id,
                        speaker='Applicant',
                        extracted_text=f"Claim {claim_num} amended to incorporate matter from cancelled claim {c_num}",
                        relevance_category='Estoppel',
                        context_summary="Zombie Claim: broader scope surrendered by incorporating dependent claim",
                        affected_claims=[claim_num],
                    )
                    session.add(stmt)
        
        version = ClaimVersion(
            claim_id=claim.id,
            document_id=doc.id,
            version_number=claim_data['version'] + 1,
            claim_text=new_text,
            normalized_text=new_normalized,
            change_summary=amend.get('change_summary', ''),
            is_normalized_change=is_substantive,
            date_of_change=section.date,
            added_limitations=amend.get('added_limitations'),
            amendment_source="applicant",  # Gap 3: Default to applicant-authored
        )
        
        # Gap 10: Check if this amendment implements an interview suggestion
        # If so, annotate the change_summary and set amendment_source
        change_summary_text = version.change_summary or ''
        if '[Implements Examiner Interview Suggestion]' in change_summary_text or \
           'incorporates examiner suggestion' in (doc.raw_text or '').lower()[:2000]:
            version.amendment_source = "examiner_interview"
        
        # Issue 4 fix: Flag likely paraphrased/summarized claim text.
        # Real claim text is typically 50-500+ words. A 30-word summary is clearly not full text.
        word_count = len(new_text.split()) if new_text else 0
        if word_count < 30 and is_substantive and new_text.strip():
            existing_summary = version.change_summary or ''
            version.change_summary = f"⚠️ [Approximate reconstruction — text may be paraphrased] {existing_summary}"
            logger.info(f"Claim {claim_num}: Flagged as possibly paraphrased ({word_count} words)")
        
        session.add(version)
        
        # Link limitations to rejections
        self._link_limitations_to_rejections(amend, active_rejections, version)
        
        self.current_claims[claim_num] = {
            'id': claim.id,
            'text': new_text,
            'version': claim_data['version'] + 1,
        }
        
        if claim:
            claim.current_status = 'Pending'
    
    def _link_limitations_to_rejections(self, amend, active_rejections, version):
        """Link added limitations to the rejections they address"""
        if not active_rejections or not amend.get('added_limitations'):
            return
        for limit in amend['added_limitations']:
            target_art = limit.get('likely_response_to_art', '')
            if target_art and target_art.lower() != 'none':
                for rej in active_rejections:
                    if target_art.lower() in str(rej.cited_prior_art).lower():
                        rej.addressed_by_version = version
                        rej.is_overcome = True
    
    def _get_prior_art_context(self, session, doc) -> str:
        """Get prior art context from the most recent office action"""
        last_oa = session.query(Document).filter(
            Document.patent_id == self.current_patent.id,
            Document.document_type == "Office Action",
            Document.document_date < (doc.document_date or datetime.now()),
        ).order_by(Document.document_date.desc()).first()
        
        if not last_oa:
            return "None identified."
        
        rejections = session.query(RejectionHistory).filter(
            RejectionHistory.document_id == last_oa.id
        ).all()
        
        if not rejections:
            return "None identified."
        
        descriptions = []
        for r in rejections:
            art_desc = "Unknown"
            if r.cited_prior_art:
                if isinstance(r.cited_prior_art, list):
                    refs = [str(a.get('reference', a) if isinstance(a, dict) else a) for a in r.cited_prior_art]
                    art_desc = ", ".join(refs)
                else:
                    art_desc = str(r.cited_prior_art)
            descriptions.append(f"- Claim {r.claim_number} ({r.statutory_basis}): {art_desc}")
        
        return "\n".join(descriptions)
    
    def _get_active_rejections(self, session, doc) -> List[RejectionHistory]:
        """Get active rejections from the most recent office action"""
        last_oa = session.query(Document).filter(
            Document.patent_id == self.current_patent.id,
            Document.document_type == "Office Action",
            Document.document_date < (doc.document_date or datetime.now()),
        ).order_by(Document.document_date.desc()).first()
        
        if not last_oa:
            return []
        
        return session.query(RejectionHistory).filter(
            RejectionHistory.document_id == last_oa.id
        ).all()
    
    def _extract_arguments(self, session, doc: Document, section: DocumentSection):
        """Extract prosecution arguments with quote verification and negative limitation detection"""
        try:
            result = self._process_with_chunking(
                text=section.text,
                prompt_template=ARGUMENT_EXTRACTION_PROMPT,
                task_type=TaskType.ARGUMENT_EXTRACTION,
                merge_func=self.result_merger.merge_statements,
            )
            
            for stmt_data in result.get('statements', []):
                # Quote verification
                if self.verification_enabled:
                    quote = stmt_data.get('extracted_text', '')
                    
                    # Issue 1 fix: Filter purely procedural statements
                    if is_procedural_statement(quote):
                        logger.info(f"Filtering procedural statement in arguments: '{quote[:60]}...'")
                        continue
                    
                    ver = self.verifier.verify_quote(section.text, quote)
                    
                    self.statement_verifier.verification_log.append({
                        'quote': quote[:100] + '...' if len(quote) > 100 else quote,
                        'status': ver.status.value,
                        'confidence': ver.confidence,
                    })
                    
                    if ver.is_suspicious:
                        logger.warning(f"Potential hallucination: '{quote[:50]}...'")
                        existing_ctx = stmt_data.get('context', '') or ''
                        stmt_data['context'] = f"[WARNING: Quote not verified] {existing_ctx}".strip()
                        if self.downgrade_unverified:
                            stmt_data['estoppel_risk'] = 'unknown'
                    
                    if ver.matched_text and ver.is_verified:
                        stmt_data['extracted_text'] = ver.matched_text
                
                # Disavowal confidence filtering
                if stmt_data.get('negative_limitation') and stmt_data.get('disavowal_confidence') == 'low':
                    logger.info(f"Filtering low-confidence disavowal: '{stmt_data.get('extracted_text', '')[:50]}...'")
                    continue  # Skip this statement entirely
                
                # Build context summary — include negative limitation flag
                context = stmt_data.get('context', '')
                if stmt_data.get('negative_limitation'):
                    if stmt_data.get('disavowal_confidence') == 'medium':
                        context = f"[UNCONFIRMED DISAVOWAL] {context}".strip()
                    else:
                        context = f"[NEGATIVE LIMITATION / DISAVOWAL] {context}".strip()
                
                # Normalize the relevance category (Gap 1 fix)
                raw_category = stmt_data.get('statement_type')
                normalized_category = normalize_category(raw_category) if raw_category else None
                
                # Filter generic claim element terms (Gap 7 fix)
                claim_element = stmt_data.get('claim_element')
                if is_generic_term(claim_element):
                    claim_element = None
                
                # Normalize cited_prior_art into a list
                cited_art = None
                if stmt_data.get('cited_prior_art'):
                    art = stmt_data['cited_prior_art']
                    if isinstance(art, list):
                        cited_art = art
                    elif isinstance(art, str) and art.lower() not in ('none', 'null', 'n/a', ''):
                        cited_art = [art]
                
                statement = ProsecutionStatement(
                    document_id=doc.id,
                    speaker=stmt_data.get('speaker', 'Applicant'),
                    extracted_text=stmt_data.get('extracted_text', ''),
                    relevance_category=normalized_category,
                    claim_element_defined=claim_element,
                    affected_claims=stmt_data.get('affected_claims'),
                    cited_prior_art=cited_art,
                    context_summary=context,
                    traversal_present=stmt_data.get('traversal_present', False),
                    is_acquiesced=stmt_data.get('speaker') == 'Examiner',
                )
                session.add(statement)
                
        except Exception as e:
            logger.error(f"Argument extraction failed: {e}")
    
    def _check_means_plus_function(self, session, doc: Document, section: DocumentSection):
        """Check for means-plus-function (112(f)) issues"""
        claims_text = json.dumps({n: c['text'] for n, c in self.current_claims.items()})
        
        try:
            result = self._process_with_chunking(
                text=section.text,
                prompt_template=MEANS_PLUS_FUNCTION_PROMPT,
                task_type=TaskType.MEANS_PLUS_FUNCTION,
                merge_func=self.result_merger.merge_means_plus_function,
                extra_context={'claims_text': claims_text},
            )
            
            for mpf in result.get('means_plus_function_elements', []):
                claim_num = mpf.get('claim_number')
                if claim_num in self.current_claims:
                    claim = session.query(Claim).get(self.current_claims[claim_num]['id'])
                    if claim:
                        claim.is_means_plus_function = True
                        existing = claim.means_plus_function_elements or []
                        existing.append(mpf)
                        claim.means_plus_function_elements = existing
                        
        except Exception as e:
            logger.error(f"Means-plus-function check failed: {e}")
    
    def _process_restriction(self, session, doc: Document, section: DocumentSection):
        """Process a Restriction Requirement"""
        logger.info("Processing Restriction Requirement")
        
        try:
            result = self._process_with_chunking(
                text=section.text,
                prompt_template=RESTRICTION_ANALYSIS_PROMPT,
                task_type=TaskType.RESTRICTION_ANALYSIS,
                merge_func=self.result_merger.merge_restriction,
            )
            
            restriction = RestrictionRequirement(
                patent_id=self.current_patent.id,
                document_id=doc.id,
                restriction_date=section.date,
                groups={g['group_name']: g['claims'] for g in result.get('groups', [])},
                elected_group=result.get('elected_group'),
                elected_claims=result.get('elected_claims'),
                non_elected_claims=result.get('non_elected_claims'),
                traversed=result.get('traversed', False),
                traversal_arguments=result.get('traversal_arguments'),
            )
            session.add(restriction)
            
            for claim_num in result.get('non_elected_claims', []):
                if claim_num in self.current_claims:
                    claim = session.query(Claim).get(self.current_claims[claim_num]['id'])
                    if claim:
                        claim.current_status = 'Withdrawn - Non-Elected'
                        claim.is_elected = False
                        claim.elected_group = result.get('elected_group')
                        
        except Exception as e:
            logger.error(f"Restriction processing failed: {e}")
    
    def _process_allowance(self, session, doc: Document, section: DocumentSection):
        """Process a Notice of Allowance"""
        logger.info("Processing Notice of Allowance")
        
        try:
            result = self._process_with_chunking(
                text=section.text,
                prompt_template=ALLOWANCE_ANALYSIS_PROMPT,
                task_type=TaskType.ALLOWANCE_ANALYSIS,
                merge_func=self.result_merger.merge_allowance,
            )
            
            # Handle renumbering
            for renum in result.get('renumbering', []):
                app_num = _safe_claim_int(renum.get('application_claim_number'))
                issued_num = _safe_claim_int(renum.get('issued_claim_number'))
                if app_num in self.current_claims:
                    claim = session.query(Claim).get(self.current_claims[app_num]['id'])
                    if claim:
                        claim.final_issued_number = issued_num
                        claim.current_status = 'Allowed'
            
            # Determine allowed claims
            allowed = result.get('allowed_claims', [])
            if self.final_claims_parsed and not allowed:
                allowed = [c.number for c in self.final_claims_parsed]
            if not allowed and "all claims" in section.text.lower():
                allowed = list(self.current_claims.keys())
            
            for claim_num in allowed:
                claim_num = _safe_claim_int(claim_num)
                if claim_num in self.current_claims:
                    claim = session.query(Claim).get(self.current_claims[claim_num]['id'])
                    if claim:
                        claim.current_status = 'Allowed'
            
            # Reasons for allowance
            rfa = result.get('reasons_for_allowance', {})
            if rfa.get('stated') and rfa.get('text'):
                stmt = ProsecutionStatement(
                    document_id=doc.id,
                    speaker='Examiner',
                    extracted_text=rfa['text'],
                    relevance_category='Reasons for Allowance',
                    is_acquiesced=True,
                    affected_claims=allowed,
                    context_summary=f"Key features: {', '.join(rfa.get('key_distinguishing_features', []))}",
                )
                session.add(stmt)
            
            # =================================================================
            # GAP 4 fix: Track examiner amendments as distinct claim versions.
            # Examiner amendments have distinct estoppel implications — they are
            # not applicant-initiated narrowing but examiner-defined scope.
            # =================================================================
            for ea in result.get('examiner_amendments', []):
                ea_claim_num = ea.get('claim_number')
                ea_text = ea.get('amendment_text', '')
                if ea_claim_num and ea_claim_num in self.current_claims and ea_text:
                    claim_data = self.current_claims[ea_claim_num]
                    claim = session.query(Claim).get(claim_data['id'])
                    
                    if claim:
                        # Create a new ClaimVersion for the examiner amendment
                        ea_version = ClaimVersion(
                            claim_id=claim.id,
                            document_id=doc.id,
                            version_number=claim_data['version'] + 1,
                            claim_text=ea_text if len(ea_text) > 50 else claim_data['text'],
                            normalized_text=normalize_text(ea_text if len(ea_text) > 50 else claim_data['text']),
                            change_summary=f"[EXAMINER AMENDMENT] {ea_text}",
                            is_normalized_change=True,
                            date_of_change=section.date,
                            amendment_source="examiner",  # Gap 3: Track examiner authorship
                        )
                        session.add(ea_version)
                        
                        self.current_claims[ea_claim_num] = {
                            'id': claim.id,
                            'text': ea_text if len(ea_text) > 50 else claim_data['text'],
                            'version': claim_data['version'] + 1,
                        }
                        
                        # Also record as a prosecution statement for estoppel tracking
                        ea_stmt = ProsecutionStatement(
                            document_id=doc.id,
                            speaker='Examiner',
                            extracted_text=f"Examiner's Amendment to Claim {ea_claim_num}: {ea_text}",
                            relevance_category='Examiner-Defined Scope',
                            is_acquiesced=True,
                            affected_claims=[ea_claim_num],
                            context_summary="Examiner amendment at allowance — defines scope by examiner, "
                                          "not applicant. Creates distinct estoppel implications.",
                        )
                        session.add(ea_stmt)
                        logger.info(f"Tracked examiner amendment for Claim {ea_claim_num} as version {claim_data['version'] + 1}")
            
            session.flush()
                
        except Exception as e:
            logger.error(f"Allowance processing failed: {e}")
    
    def _process_interview(self, session, doc: Document, section: DocumentSection):
        """Process an Interview Summary"""
        logger.info("Processing Interview Summary")
        
        try:
            result = self._process_with_chunking(
                text=section.text,
                prompt_template=INTERVIEW_SUMMARY_PROMPT,
                task_type=TaskType.INTERVIEW_ANALYSIS,
                merge_func=self.result_merger.merge_interview,
            )
            
            # Store suggestions for handshake with next amendment
            # Fix #7: Also store proposed_amendments as pending suggestions
            if result.get('examiner_suggestions'):
                self.pending_interview_suggestions.extend(result['examiner_suggestions'])
            if result.get('proposed_amendments'):
                self.pending_interview_suggestions.extend(result['proposed_amendments'])
            
            # Fix #7: Record examiner claim constructions from interview
            for construction in result.get('examiner_claim_constructions', []):
                if construction and len(construction) > 5:
                    stmt = ProsecutionStatement(
                        document_id=doc.id,
                        speaker='Examiner',
                        extracted_text=construction,
                        relevance_category='Definition/Interpretation',
                        is_acquiesced=True,
                        context_summary="Examiner claim construction stated during interview",
                    )
                    session.add(stmt)
            
            # Record agreements
            for agreement in result.get('agreements_reached', []):
                stmt = ProsecutionStatement(
                    document_id=doc.id,
                    speaker='Applicant/Examiner',
                    extracted_text=f"Agreement on {agreement.get('topic', 'unknown topic')}",
                    relevance_category='Interview Concession',
                    is_acquiesced=True,
                    affected_claims=agreement.get('claims_affected'),
                    context_summary=f"Interview Agreement. Estoppel risk: {agreement.get('potential_estoppel')}",
                )
                session.add(stmt)
                
        except Exception as e:
            logger.error(f"Interview processing failed: {e}")
        
        # Also extract arguments
        self._extract_arguments(session, doc, section)
    
    def _process_terminal_disclaimer(self, session, doc: Document, section: DocumentSection):
        """Process a Terminal Disclaimer"""
        logger.info("Processing Terminal Disclaimer")
        
        try:
            result = self._process_with_chunking(
                text=section.text,
                prompt_template=TERMINAL_DISCLAIMER_PROMPT,
                task_type=TaskType.TERMINAL_DISCLAIMER,
                merge_func=self.result_merger.merge_terminal_disclaimer,
            )
            
            if result.get('has_terminal_disclaimer'):
                for disc in result.get('disclaimed_patents', []):
                    td = TerminalDisclaimer(
                        patent_id=self.current_patent.id,
                        document_id=doc.id,
                        disclaimed_patent=disc.get('patent_number'),
                        disclaimer_date=section.date,
                        reason=result.get('reason'),
                    )
                    session.add(td)
                    
        except Exception as e:
            logger.error(f"Terminal disclaimer processing failed: {e}")
    
    def _process_appeal_brief(self, session, doc: Document, section: DocumentSection):
        """Process an Appeal Brief"""
        logger.info("Processing Appeal Brief")
        self._extract_arguments(session, doc, section)
        self._check_means_plus_function(session, doc, section)
    
    def _process_ptab_decision(self, session, doc: Document, section: DocumentSection):
        """Process a PTAB Decision"""
        logger.info("Processing PTAB Decision")
        self._extract_arguments(session, doc, section)
    
    def _process_reasons_for_allowance(self, session, doc: Document, section: DocumentSection):
        """Process Examiner's Reasons for Allowance"""
        logger.info("Processing Reasons for Allowance")
        
        stmt = ProsecutionStatement(
            document_id=doc.id,
            speaker='Examiner',
            extracted_text=section.text[:5000],
            relevance_category='Reasons for Allowance',
            is_acquiesced=True,
            context_summary='Examiner Statement of Reasons for Allowance',
        )
        session.add(stmt)
    
    def _process_generic(self, session, doc: Document, section: DocumentSection):
        """Generic processing for unhandled document types"""
        logger.info(f"Generic processing: {section.document_type}")
        
        prompt_with_type = COMPREHENSIVE_ANALYSIS_PROMPT.replace(
            "Provided Document Type: {document_type}",
            f"Provided Document Type: {section.document_type}",
        )
        
        try:
            result = self._process_with_chunking(
                text=section.text,
                prompt_template=prompt_with_type,
                task_type=TaskType.GENERIC_ANALYSIS,
                merge_func=self.result_merger.merge_comprehensive,
            )
            
            # Update document type if AI found a better one
            new_type = result.get('document_type') or result.get('corrected_document_type')
            if new_type and new_type != "Unknown":
                doc.document_type = new_type
            
            # Update date if found
            if result.get('document_date'):
                try:
                    doc.document_date = datetime.strptime(result['document_date'], "%Y-%m-%d")
                except (ValueError, TypeError):
                    pass
            
            # Handle claims
            if result.get('claims', {}).get('present'):
                for claim_data in result['claims'].get('claim_list', []):
                    claim_num = _safe_claim_int(claim_data.get('number'))
                    if claim_num and claim_num not in self.current_claims:
                        claim = Claim(
                            patent_id=self.current_patent.id,
                            application_claim_number=claim_num,
                            is_independent=claim_data.get('is_independent', True),
                            current_status='Pending',
                        )
                        session.add(claim)
                        session.flush()
                        
                        version = ClaimVersion(
                            claim_id=claim.id,
                            document_id=doc.id,
                            version_number=1,
                            claim_text=claim_data.get('text') or '',
                            date_of_change=section.date,
                        )
                        session.add(version)
                        
                        self.current_claims[claim_num] = {
                            'id': claim.id,
                            'text': claim_data.get('text') or '',
                            'version': 1,
                        }
            
            # Handle key statements
            for stmt_data in result.get('key_statements', []):
                stmt = ProsecutionStatement(
                    document_id=doc.id,
                    speaker=stmt_data.get('speaker', 'Unknown'),
                    extracted_text=stmt_data.get('text', ''),
                    context_summary=stmt_data.get('significance'),
                    is_acquiesced=stmt_data.get('speaker') == 'Examiner',
                )
                session.add(stmt)
                
        except Exception as e:
            logger.error(f"Generic processing failed: {e}")
    
    # =========================================================================
    # POST-PROCESSING: LAYER 2/3 SYNTHESIS & CRITIQUE PIPELINE
    # =========================================================================
    
    def _post_process_data(self, session):
        """
        Run post-processing pipeline after all raw extraction is complete.
        
        This is the "Analyst Layer" described in the v2.1 design doc.
        It queries committed raw data and generates derived intelligence
        stored in additive tables (never modifies raw data).
        
        Pipeline:
          A. Aggregate raw data by term and by claim
          B. Run AI analysis (Shadow Examiner, Definition Synthesis, Claim Narratives)
          C. Store results in new tables
        """
        patent_id = self.current_patent.id
        pp_config = self.config.get('post_processing', {})
        
        # Clear any previous post-processing results for this patent
        # (safe to re-run without duplicating)
        session.query(ValidityRisk).filter(ValidityRisk.patent_id == patent_id).delete()
        session.query(TermSynthesis).filter(TermSynthesis.patent_id == patent_id).delete()
        session.query(ClaimNarrative).filter(ClaimNarrative.patent_id == patent_id).delete()
        session.query(PatentTheme).filter(PatentTheme.patent_id == patent_id).delete()
        session.query(TermBoundary).filter(TermBoundary.patent_id == patent_id).delete()
        session.query(PriorArtReference).filter(PriorArtReference.patent_id == patent_id).delete()
        session.query(ClaimVulnerabilityCard).filter(ClaimVulnerabilityCard.patent_id == patent_id).delete()
        session.flush()
        
        # --- A. Shadow Examiner (Validity Critique) ---
        if pp_config.get('shadow_examiner', True):
            try:
                self._run_shadow_examiner(session, patent_id)
            except Exception as e:
                logger.error(f"Shadow Examiner failed: {e}")
        
        # --- B. Definition Synthesis & Consistency Check ---
        if pp_config.get('definition_synthesis', True):
            try:
                self._run_definition_synthesis(session, patent_id)
            except Exception as e:
                logger.error(f"Definition Synthesis failed: {e}")
        
        # --- B2. Term Boundary Extraction (Gap 1) ---
        if pp_config.get('term_boundaries', True):
            try:
                self._run_term_boundary_extraction(session, patent_id)
            except Exception as e:
                logger.error(f"Term Boundary Extraction failed: {e}")
        
        # --- C. Claim Narratives (Biographies) ---
        if pp_config.get('claim_narratives', True):
            try:
                self._run_claim_narratives(session, patent_id)
            except Exception as e:
                logger.error(f"Claim Narratives failed: {e}")
        
        # --- D. Strategic Tensions / Contradictions Detection (Gap 1 fix) ---
        if pp_config.get('strategic_tensions', True):
            try:
                self._run_strategic_tensions(session, patent_id)
            except Exception as e:
                logger.error(f"Strategic Tensions detection failed: {e}")
        
        # --- E. Thematic Synthesis (Improvement A) ---
        if pp_config.get('thematic_synthesis', True):
            try:
                self._run_thematic_synthesis(session, patent_id)
            except Exception as e:
                logger.error(f"Thematic Synthesis failed: {e}")
        
        # --- F. Prior Art Reference Consolidation (Gap 4) ---
        if pp_config.get('prior_art_consolidation', True):
            try:
                self._consolidate_prior_art_references(session, patent_id)
            except Exception as e:
                logger.error(f"Prior Art Consolidation failed: {e}")
        
        # --- G. Claim Type Classification (Gap 12) ---
        if pp_config.get('claim_type_classification', True):
            try:
                self._classify_final_claim_types(session, patent_id)
            except Exception as e:
                logger.error(f"Claim Type Classification failed: {e}")
        
        # --- H. Per-Claim Vulnerability Cards (Gap 5) ---
        if pp_config.get('vulnerability_cards', True):
            try:
                self._run_vulnerability_cards(session, patent_id)
            except Exception as e:
                logger.error(f"Vulnerability Cards failed: {e}")
        
        session.flush()
    
    def _run_shadow_examiner(self, session, patent_id: str):
        """
        Task 13: Shadow Examiner — critique allowed claims for validity risks.
        
        Only runs if there are allowed/final claims to analyze.
        """
        # Gather claim text: prefer final claims, fall back to latest versions of allowed claims
        claims_text_parts = []
        
        final_claims = session.query(FinalClaim).filter(
            FinalClaim.patent_id == patent_id
        ).order_by(FinalClaim.claim_number).all()
        
        if final_claims:
            for fc in final_claims:
                dep_info = f" (depends on claim {fc.depends_on})" if fc.depends_on else " (independent)"
                claims_text_parts.append(f"Claim {fc.claim_number}{dep_info}:\n{fc.claim_text}")
        else:
            # Fall back to allowed application claims
            allowed_claims = session.query(Claim).filter(
                Claim.patent_id == patent_id,
                Claim.current_status == 'Allowed',
            ).all()
            
            for claim in allowed_claims:
                latest = session.query(ClaimVersion).filter(
                    ClaimVersion.claim_id == claim.id,
                ).order_by(ClaimVersion.version_number.desc()).first()
                if latest:
                    claims_text_parts.append(
                        f"Claim {claim.application_claim_number}:\n{latest.claim_text}"
                    )
        
        if not claims_text_parts:
            logger.info("Shadow Examiner: No allowed/final claims to analyze — skipping")
            return
        
        claims_text = "\n\n".join(claims_text_parts)
        
        # Gather prosecution context (key arguments/estoppel events)
        statements = session.query(ProsecutionStatement).join(Document).filter(
            Document.patent_id == patent_id,
            ProsecutionStatement.relevance_category.in_([
                'Prior Art Distinction', 'Estoppel', 'scope_limitation',
                'definition', 'disavowal', 'concession',
            ]),
        ).order_by(Document.document_date).limit(20).all()
        
        prosecution_context = "\n".join(
            f"- [{s.speaker}] {s.extracted_text[:300]}"
            for s in statements
        ) or "No significant prosecution arguments recorded."
        
        prompt = SHADOW_EXAMINER_PROMPT.format(
            claims_text=claims_text[:15000],
            prosecution_context=prosecution_context[:5000],
        )
        
        provider = self._get_provider(TaskType.SHADOW_EXAMINER)
        result = provider.complete_json(prompt, SYSTEM_PROMPT)
        
        min_severity = self.config.get('post_processing', {}).get(
            'shadow_examiner_min_severity', 'Medium'
        )
        severity_rank = {'High': 3, 'Medium': 2, 'Low': 1}
        min_rank = severity_rank.get(min_severity, 2)
        
        for risk in result.get('risks', []):
            risk_severity = risk.get('severity', 'Medium')
            if severity_rank.get(risk_severity, 2) < min_rank:
                continue
            
            # Verify offending_text_quote against actual claims text
            offending_quote = risk.get('offending_text_quote', '')
            description = risk.get('description', '')
            reasoning = risk.get('reasoning') or ''
            
            if offending_quote and self.verification_enabled:
                ver = self.verifier.verify_quote(claims_text, offending_quote)
                if ver.is_suspicious:
                    logger.warning(f"Shadow Examiner: offending_text_quote not verified: '{offending_quote[:50]}...'")
                    reasoning = f"[WARNING: Quoted claim text not verified in source] {reasoning}".strip()
                    if self.downgrade_unverified:
                        risk_severity = 'Low'
            
            # Prepend offending quote to description for traceability
            if offending_quote:
                description = f'[Claim text: "{offending_quote}"] {description}'.strip()
            
            vr = ValidityRisk(
                patent_id=patent_id,
                risk_type=risk.get('type', 'Unknown'),
                severity=risk_severity,
                description=description,
                reasoning=reasoning,
                affected_claims=risk.get('claims'),
            )
            session.add(vr)
        
        logger.info(f"Shadow Examiner: {len(result.get('risks', []))} risks identified")
    
    def _run_definition_synthesis(self, session, patent_id: str):
        """
        Task 14: Synthesize definitions and check for flip-flops.
        
        Groups ProsecutionStatements by claim_element_defined, then sends
        each group to the LLM for synthesis and consistency analysis.
        Only synthesizes terms with >1 statement (per design doc token budget guidance).
        """
        # Query all statements that define a claim element
        statements = session.query(ProsecutionStatement).join(Document).filter(
            Document.patent_id == patent_id,
            ProsecutionStatement.claim_element_defined.isnot(None),
            ProsecutionStatement.claim_element_defined != '',
        ).order_by(Document.document_date).all()
        
        if not statements:
            logger.info("Definition Synthesis: No defined terms found — skipping")
            return
        
        # Group by term
        terms: Dict[str, List[ProsecutionStatement]] = {}
        for stmt in statements:
            term = stmt.claim_element_defined.strip()
            if term:
                terms.setdefault(term, []).append(stmt)
        
        min_statements = self.config.get('post_processing', {}).get(
            'synthesis_min_statements', 1
        )
        
        provider = self._get_provider(TaskType.DEFINITION_SYNTHESIS)
        synthesized_count = 0
        
        for term, stmts in terms.items():
            if len(stmts) < min_statements:
                continue
            
            # Build chronological statements list for the prompt
            statements_list = "\n".join(
                f"[{i+1}] ({s.document.document_date.strftime('%Y-%m-%d') if s.document.document_date else 'Unknown date'}) "
                f"[{s.speaker}]: \"{s.extracted_text[:500]}\""
                for i, s in enumerate(stmts)
            )
            
            prompt = DEFINITION_SYNTHESIS_PROMPT.format(
                term=term,
                statements_list=statements_list,
            )
            
            try:
                result = provider.complete_json(prompt, SYSTEM_PROMPT)
                
                ts = TermSynthesis(
                    patent_id=patent_id,
                    term=result.get('term', term),
                    narrative_summary=result.get('narrative_summary', ''),
                    consistency_status=result.get('consistency_status', 'Unknown'),
                    contradiction_details=result.get('contradiction_analysis'),
                    source_statement_ids=[s.id for s in stmts],
                )
                session.add(ts)
                synthesized_count += 1
                
            except Exception as e:
                logger.error(f"Definition synthesis failed for '{term}': {e}")
        
        logger.info(f"Definition Synthesis: {synthesized_count} terms synthesized from {len(terms)} total")
    
    def _run_claim_narratives(self, session, patent_id: str):
        """
        Task 15: Generate claim biographies for key claims.
        
        Builds a narrative from claim versions, rejections, and arguments.
        
        Prioritizes claims that map to final issued claims (Gap 4 fix),
        and includes both independent and key dependent claims.
        """
        all_claims = session.query(Claim).filter(
            Claim.patent_id == patent_id,
        ).all()
        
        if not all_claims:
            logger.info("Claim Narratives: No claims found — skipping")
            return
        
        max_narratives = self.config.get('post_processing', {}).get(
            'max_claim_narratives', 10
        )
        
        # =====================================================================
        # Gap 4 fix: Prioritize claims that map to final issued claims,
        # then other independent claims. Include key dependent claims too.
        # =====================================================================
        issued_claims = [c for c in all_claims if c.mapped_final_claim_number is not None]
        other_independent = [c for c in all_claims
                            if c.is_independent
                            and c.mapped_final_claim_number is None
                            and c.current_status in ('Allowed', 'Pending', 'Rejected')]
        
        # Sort each group by application claim number
        issued_claims.sort(key=lambda c: c.application_claim_number)
        other_independent.sort(key=lambda c: c.application_claim_number)
        
        claims_to_narrate = issued_claims + other_independent
        claims_to_narrate = claims_to_narrate[:max_narratives]
        
        # =====================================================================
        # Gap 8 fix: Pre-fetch Reasons for Allowance for all claims
        # =====================================================================
        rfa_statements = session.query(ProsecutionStatement).join(Document).filter(
            Document.patent_id == patent_id,
            ProsecutionStatement.relevance_category == 'Reasons for Allowance',
        ).all()
        rfa_text = "\n".join(
            f"- {s.extracted_text[:500]}" for s in rfa_statements
        ) if rfa_statements else "No Reasons for Allowance recorded."
        
        provider = self._get_provider(TaskType.CLAIM_NARRATIVE)
        narrative_count = 0
        
        for claim in claims_to_narrate:
            claim_num = claim.application_claim_number
            
            # Gather versions
            versions = session.query(ClaimVersion).filter(
                ClaimVersion.claim_id == claim.id,
            ).order_by(ClaimVersion.version_number).all()
            
            if not versions:
                continue
            
            versions_text = "\n".join(
                f"Version {v.version_number} "
                f"({v.date_of_change.strftime('%Y-%m-%d') if v.date_of_change else 'Unknown date'}): "
                f"{v.change_summary or '(no summary)'}\n"
                f"Text: {v.claim_text[:500]}"
                for v in versions
            )
            
            # Gather rejections affecting this claim
            rejections = session.query(RejectionHistory).join(Document).filter(
                Document.patent_id == patent_id,
                RejectionHistory.claim_number == claim_num,
            ).order_by(Document.document_date).all()
            
            rejections_text = "\n".join(
                f"- {r.statutory_basis or 'Unknown'} rejection: {r.rejection_rationale[:200] if r.rejection_rationale else 'N/A'} "
                f"(Overcome: {r.is_overcome})"
                for r in rejections
            ) or "No rejections recorded for this claim."
            
            # Gather arguments related to this claim
            arguments = session.query(ProsecutionStatement).join(Document).filter(
                Document.patent_id == patent_id,
            ).all()
            
            # Filter to those affecting this claim number
            relevant_args = [
                a for a in arguments
                if a.affected_claims and claim_num in a.affected_claims
            ][:10]
            
            arguments_text = "\n".join(
                f"- [{a.speaker}] {a.extracted_text[:200]}"
                for a in relevant_args
            ) or "No specific arguments recorded for this claim."
            
            # Fix #14: Determine dependency info for smarter dependent claim narratives
            is_dependent = not claim.is_independent
            parent_claim_num = 'N/A'
            if is_dependent:
                # Find parent claim from version text or claim dependencies
                first_version_text = versions[0].claim_text if versions else ''
                dep_match = re.search(r'claim\s+(\d+)', first_version_text, re.IGNORECASE)
                if dep_match:
                    parent_claim_num = dep_match.group(1)
            
            prompt = CLAIM_NARRATIVE_PROMPT.format(
                claim_number=claim_num,
                versions_text=versions_text[:5000],
                rejections_text=rejections_text[:3000],
                arguments_text=arguments_text[:3000],
                reasons_for_allowance_text=rfa_text[:3000],
                is_dependent=str(is_dependent),
                parent_claim_number=parent_claim_num,
            )
            
            try:
                result = provider.complete_json(prompt, SYSTEM_PROMPT)
                
                cn = ClaimNarrative(
                    patent_id=patent_id,
                    claim_number=result.get('claim_number', claim_num),
                    evolution_summary=result.get('evolution_summary', ''),
                    turning_point_event=result.get('turning_point_event'),
                )
                session.add(cn)
                narrative_count += 1
                
            except Exception as e:
                logger.error(f"Claim narrative failed for claim {claim_num}: {e}")
        
        logger.info(f"Claim Narratives: {narrative_count} narratives generated")
    
    def _run_strategic_tensions(self, session, patent_id: str):
        """
        Task 17 (Gap 1 fix): Detect strategic tensions / contradictions in prosecution history.
        
        Cross-references arguments distinguishing references with subsequent amendments
        adding the same feature. For example: arguing "no geographic map" to distinguish
        Kato, then later adding "geographic map" to distinguish Graves/Davis.
        """
        # Gather all prosecution statements and amendment limitations chronologically
        statements = session.query(ProsecutionStatement).join(Document).filter(
            Document.patent_id == patent_id,
        ).order_by(Document.document_date).all()
        
        # Gather amendment limitations from claim versions
        claims = session.query(Claim).filter(Claim.patent_id == patent_id).all()
        amendment_events = []
        
        for claim in claims:
            versions = session.query(ClaimVersion).filter(
                ClaimVersion.claim_id == claim.id,
            ).order_by(ClaimVersion.version_number).all()
            
            for v in versions:
                if v.added_limitations:
                    for lim in v.added_limitations:
                        amendment_events.append({
                            'type': 'amendment',
                            'date': v.date_of_change.strftime('%Y-%m-%d') if v.date_of_change else 'Unknown',
                            'claim': claim.application_claim_number,
                            'text': lim.get('text', ''),
                            'art': lim.get('likely_response_to_art', ''),
                            'argument': lim.get('applicant_argument_summary', ''),
                        })
        
        # Build chronological prosecution events for the prompt
        # Gap 2 fix: Use FULL text for arguments (not truncated to 300 chars)
        # and prioritize completeness over breadth
        events_text_parts = []
        for stmt in statements:
            cat = stmt.relevance_category or ''
            if cat in ('Prior Art Distinction', 'Estoppel', 'Traversal', 'General Argument'):
                date = stmt.document.document_date.strftime('%Y-%m-%d') if stmt.document and stmt.document.document_date else 'Unknown'
                art = ', '.join(stmt.cited_prior_art) if stmt.cited_prior_art else 'N/A'
                # Gap 2: Use full extracted_text (not [:300]) for tension detection
                full_text = stmt.extracted_text or ''
                events_text_parts.append(
                    f"[{date}] ARGUMENT: {full_text} (Distinguishing: {art}, Claims: {stmt.affected_claims})"
                )
        
        for ae in sorted(amendment_events, key=lambda x: x['date']):
            events_text_parts.append(
                f"[{ae['date']}] AMENDMENT to Claim {ae['claim']}: Added \"{ae['text']}\" "
                f"(Responding to: {ae['art']}; Argument: {ae['argument']})"
            )
        
        if len(events_text_parts) < 3:
            logger.info("Strategic Tensions: Not enough prosecution events to analyze — skipping")
            return
        
        # Sort by date
        events_text_parts.sort()
        
        # Gap 2: Reduce event count but keep full text to stay within token limits
        # Prioritize arguments with cited prior art (most likely to create tensions)
        max_events = 60  # Fewer events but with full text instead of 100 truncated
        
        prompt = STRATEGIC_TENSIONS_PROMPT.format(
            prosecution_events="\n".join(events_text_parts[:max_events])
        )
        
        provider = self._get_provider(TaskType.STRATEGIC_TENSIONS)
        result = provider.complete_json(prompt, SYSTEM_PROMPT)
        
        tensions = result.get('strategic_tensions', [])
        tension_count = 0
        
        for tension in tensions:
            # Store as a ProsecutionStatement with special category
            stmt = ProsecutionStatement(
                document_id=None,  # Cross-cutting — not tied to a single document
                speaker='Analysis',
                extracted_text=(
                    f"STRATEGIC TENSION: {tension.get('feature_or_concept', 'Unknown feature')} — "
                    f"First: {tension.get('first_position', '')} | "
                    f"Later: {tension.get('second_position', '')} | "
                    f"Exploitation: {tension.get('tension_explanation', '')}"
                ),
                relevance_category='Strategic Tension',
                affected_claims=tension.get('affected_claims'),
                context_summary=(
                    f"[STRATEGIC TENSION - {tension.get('severity', 'Medium')}] "
                    f"Feature: {tension.get('feature_or_concept', '')}. "
                    f"The applicant argued both sides of this feature at different points in prosecution."
                ),
            )
            # Find any document to attach to (needed for FK constraint)
            any_doc = session.query(Document).filter(Document.patent_id == patent_id).first()
            if any_doc:
                stmt.document_id = any_doc.id
                session.add(stmt)
                tension_count += 1
        
        logger.info(f"Strategic Tensions: {tension_count} tensions detected from {len(events_text_parts)} events")
    
    def _run_thematic_synthesis(self, session, patent_id: str):
        """
        Task 20 (Improvement A): Group prosecution arguments and amendments into
        overarching "Themes of Patentability".
        
        Analyzes the full set of arguments and amendments to identify 3-7 recurring
        conceptual threads (e.g., "Integrity of Original Images", "Two-Step Pose
        Determination") rather than organizing by document or reference.
        """
        # Gather all prosecution arguments chronologically
        statements = session.query(ProsecutionStatement).join(Document).filter(
            Document.patent_id == patent_id,
        ).order_by(Document.document_date).all()
        
        arguments_parts = []
        for stmt in statements:
            cat = stmt.relevance_category or ''
            if cat in ('Prior Art Distinction', 'Estoppel', 'Traversal', 'General Argument',
                       'Definition/Interpretation', 'Claim Construction', 'Strategic Tension'):
                date = stmt.document.document_date.strftime('%Y-%m-%d') if stmt.document and stmt.document.document_date else 'Unknown'
                art = ', '.join(stmt.cited_prior_art) if stmt.cited_prior_art else 'N/A'
                element = stmt.claim_element_defined or 'N/A'
                arguments_parts.append(
                    f"[{date}] ({cat}) Re: {element} | Distinguishing: {art} | "
                    f"Claims: {stmt.affected_claims} | {stmt.extracted_text[:400]}"
                )
        
        # Gather amendment limitations from claim versions
        claims = session.query(Claim).filter(Claim.patent_id == patent_id).all()
        amendments_parts = []
        
        for claim in claims:
            versions = session.query(ClaimVersion).filter(
                ClaimVersion.claim_id == claim.id,
            ).order_by(ClaimVersion.version_number).all()
            
            for v in versions:
                if v.added_limitations:
                    date = v.date_of_change.strftime('%Y-%m-%d') if v.date_of_change else 'Unknown'
                    for lim in v.added_limitations:
                        art = lim.get('likely_response_to_art', 'N/A')
                        arg = lim.get('applicant_argument_summary', '')
                        amendments_parts.append(
                            f"[{date}] Claim {claim.application_claim_number}: "
                            f"Added \"{lim.get('text', '')}\" | Art: {art} | Argument: {arg[:200]}"
                        )
        
        if len(arguments_parts) + len(amendments_parts) < 3:
            logger.info("Thematic Synthesis: Not enough prosecution data — skipping")
            return
        
        # Gather Reasons for Allowance
        rfa_parts = []
        rfa_statements = [s for s in statements if s.relevance_category == 'Reasons for Allowance']
        for stmt in rfa_statements:
            rfa_parts.append(stmt.extracted_text[:500])
        
        prompt = THEMATIC_SYNTHESIS_PROMPT.format(
            arguments_text="\n".join(arguments_parts[:80]),
            amendments_text="\n".join(amendments_parts[:50]),
            reasons_for_allowance="\n".join(rfa_parts) if rfa_parts else "Not available.",
        )
        
        provider = self._get_provider(TaskType.THEMATIC_SYNTHESIS)
        result = provider.complete_json(prompt, SYSTEM_PROMPT)
        
        themes = result.get('themes', [])
        theme_count = 0
        
        for theme_data in themes:
            theme = PatentTheme(
                patent_id=patent_id,
                title=theme_data.get('title', 'Untitled Theme'),
                summary=theme_data.get('summary', ''),
                key_arguments=theme_data.get('key_arguments', []),
                key_amendments=theme_data.get('key_amendments', []),
                prior_art_distinguished=theme_data.get('prior_art_distinguished', []),
                affected_claims=theme_data.get('affected_claims', []),
                estoppel_significance=theme_data.get('estoppel_significance', ''),
                metaphors_or_analogies=theme_data.get('metaphors_or_analogies', []),
            )
            session.add(theme)
            theme_count += 1
        
        logger.info(f"Thematic Synthesis: {theme_count} themes identified")
    
    def _run_term_boundary_extraction(self, session, patent_id: str):
        """
        Gap 1: Extract specific enumerated examples of what falls inside/outside
        prosecution-defined terms' scope.
        
        Runs AFTER definition synthesis so it can leverage synthesized terms.
        """
        # Get all synthesized terms
        term_syntheses = session.query(TermSynthesis).filter(
            TermSynthesis.patent_id == patent_id
        ).all()
        
        if not term_syntheses:
            logger.info("Term Boundary Extraction: No synthesized terms — skipping")
            return
        
        provider = self._get_provider(TaskType.TERM_BOUNDARIES)
        boundary_count = 0
        
        for ts in term_syntheses:
            # Get the original statements for this term
            stmt_ids = ts.source_statement_ids or []
            if not stmt_ids:
                continue
            
            stmts = session.query(ProsecutionStatement).filter(
                ProsecutionStatement.id.in_(stmt_ids)
            ).order_by(ProsecutionStatement.created_at).all()
            
            if not stmts:
                continue
            
            statements_list = "\n".join(
                f"[{i+1}] ({s.document.document_date.strftime('%Y-%m-%d') if s.document and s.document.document_date else 'Unknown'}) "
                f"[{s.speaker}]: \"{s.extracted_text[:800]}\""
                for i, s in enumerate(stmts)
            )
            
            prompt = TERM_BOUNDARY_EXTRACTION_PROMPT.format(
                term=ts.term,
                statements_list=statements_list,
            )
            
            try:
                result = provider.complete_json(prompt, SYSTEM_PROMPT)
                
                for boundary in result.get('boundaries', []):
                    if boundary.get('confidence', 'medium') == 'low':
                        continue  # Skip low-confidence boundaries
                    
                    tb = TermBoundary(
                        patent_id=patent_id,
                        term_synthesis_id=ts.id,
                        term=ts.term,
                        boundary_type=boundary.get('boundary_type', 'excludes'),
                        example_text=boundary.get('example_text', ''),
                        source_text=boundary.get('source_quote', ''),
                        affected_claims=stmts[0].affected_claims if stmts else None,
                    )
                    session.add(tb)
                    boundary_count += 1
                    
            except Exception as e:
                logger.error(f"Term boundary extraction failed for '{ts.term}': {e}")
        
        logger.info(f"Term Boundary Extraction: {boundary_count} boundaries extracted from {len(term_syntheses)} terms")
    
    def _consolidate_prior_art_references(self, session, patent_id: str):
        """
        Gap 4: Consolidate scattered prior art references into a normalized,
        deduplicated PriorArtReference table.
        """
        from difflib import get_close_matches
        
        # Gather all prior art mentions from rejections
        rejections = session.query(RejectionHistory).join(Document).filter(
            Document.patent_id == patent_id
        ).all()
        
        # Gather from prosecution statements
        statements = session.query(ProsecutionStatement).join(Document).filter(
            Document.patent_id == patent_id,
            ProsecutionStatement.cited_prior_art.isnot(None),
        ).all()
        
        # Build a consolidated map: canonical_name -> data
        ref_data = {}  # name -> {patent_number, bases, claims, teachings, deficiencies, overcome}
        
        for rej in rejections:
            if not rej.cited_prior_art:
                continue
            arts = rej.cited_prior_art if isinstance(rej.cited_prior_art, list) else [rej.cited_prior_art]
            for art in arts:
                if isinstance(art, dict):
                    name = art.get('reference', 'Unknown')
                    pat_num = art.get('patent_or_pub_number')
                else:
                    name = str(art)
                    pat_num = None
                
                if not name or name.lower() in ('none', 'null', 'n/a', ''):
                    continue
                
                name_key = name.strip()
                if name_key not in ref_data:
                    ref_data[name_key] = {
                        'patent_number': pat_num,
                        'bases': set(),
                        'claims': set(),
                        'teachings': [],
                        'deficiencies': [],
                        'overcome': rej.is_overcome,
                    }
                
                if rej.statutory_basis:
                    ref_data[name_key]['bases'].add(rej.statutory_basis)
                ref_data[name_key]['claims'].add(rej.claim_number)
                if rej.rejection_rationale:
                    ref_data[name_key]['teachings'].append(rej.rejection_rationale[:200])
                if pat_num and not ref_data[name_key]['patent_number']:
                    ref_data[name_key]['patent_number'] = pat_num
        
        for stmt in statements:
            if not stmt.cited_prior_art:
                continue
            for art_name in stmt.cited_prior_art:
                if not art_name or art_name.lower() in ('none', 'null', 'n/a', ''):
                    continue
                name_key = art_name.strip()
                if name_key not in ref_data:
                    ref_data[name_key] = {
                        'patent_number': None,
                        'bases': set(),
                        'claims': set(),
                        'teachings': [],
                        'deficiencies': [],
                        'overcome': False,
                    }
                if stmt.affected_claims:
                    ref_data[name_key]['claims'].update(stmt.affected_claims)
                # Applicant statements about what art lacks = deficiencies
                if stmt.speaker == 'Applicant' and stmt.extracted_text:
                    ref_data[name_key]['deficiencies'].append(stmt.extracted_text[:200])
        
        # Fuzzy-merge similar reference names (e.g., "Kato" and "Kato et al.")
        all_names = list(ref_data.keys())
        canonical_map = {}
        processed = set()
        
        for name in sorted(all_names, key=len):
            if name in processed:
                continue
            canonical_map[name] = name
            processed.add(name)
            
            others = [n for n in all_names if n not in processed]
            matches = get_close_matches(name.lower(), [n.lower() for n in others], n=3, cutoff=0.75)
            for match_lower in matches:
                for other_name in others:
                    if other_name.lower() == match_lower and other_name not in processed:
                        canonical_map[other_name] = name
                        processed.add(other_name)
        
        # Store consolidated references
        stored_count = 0
        stored_canonicals = set()
        
        for orig_name, data in ref_data.items():
            canonical = canonical_map.get(orig_name, orig_name)
            if canonical in stored_canonicals:
                continue
            stored_canonicals.add(canonical)
            
            # Merge data from all names that map to this canonical
            merged_bases = set()
            merged_claims = set()
            merged_teachings = []
            merged_deficiencies = []
            pat_num = None
            is_overcome = False
            
            for name, cname in canonical_map.items():
                if cname == canonical and name in ref_data:
                    d = ref_data[name]
                    merged_bases.update(d['bases'])
                    merged_claims.update(d['claims'])
                    merged_teachings.extend(d['teachings'])
                    merged_deficiencies.extend(d['deficiencies'])
                    if d['patent_number']:
                        pat_num = d['patent_number']
                    if d['overcome']:
                        is_overcome = True
            
            par = PriorArtReference(
                patent_id=patent_id,
                canonical_name=canonical,
                patent_or_pub_number=pat_num,
                applied_basis=list(merged_bases) if merged_bases else None,
                affected_claims=sorted(merged_claims, key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else float('inf')) if merged_claims else None,
                key_teachings=merged_teachings[:5] if merged_teachings else None,
                key_deficiencies=merged_deficiencies[:5] if merged_deficiencies else None,
                is_overcome=is_overcome,
            )
            session.add(par)
            stored_count += 1
        
        logger.info(f"Prior Art Consolidation: {stored_count} references from {len(ref_data)} raw entries")
    
    def _classify_final_claim_types(self, session, patent_id: str):
        """
        Gap 12: Classify each final claim's statutory category from its preamble text.
        """
        final_claims = session.query(FinalClaim).filter(
            FinalClaim.patent_id == patent_id
        ).all()
        
        for fc in final_claims:
            fc.claim_type = classify_claim_type(fc.claim_text)
        
        if final_claims:
            logger.info(f"Claim Type Classification: Classified {len(final_claims)} final claims")
    
    def _run_vulnerability_cards(self, session, patent_id: str):
        """
        Gap 5: Generate per-claim vulnerability cards for independent claims.
        Synthesizes estoppel events, validity risks, and prosecution positions
        into a single per-claim view.
        """
        final_claims = session.query(FinalClaim).filter(
            FinalClaim.patent_id == patent_id,
            FinalClaim.is_independent == True,
        ).order_by(FinalClaim.claim_number).all()
        
        if not final_claims:
            logger.info("Vulnerability Cards: No independent final claims — skipping")
            return
        
        # Pre-fetch all relevant data
        all_statements = session.query(ProsecutionStatement).join(Document).filter(
            Document.patent_id == patent_id,
        ).all()
        
        all_risks = session.query(ValidityRisk).filter(
            ValidityRisk.patent_id == patent_id,
        ).all()
        
        all_claims = session.query(Claim).filter(
            Claim.patent_id == patent_id,
        ).all()
        
        provider = self._get_provider(TaskType.VULNERABILITY_CARDS)
        card_count = 0
        
        for fc in final_claims[:10]:  # Limit to 10 independent claims
            claim_num = fc.claim_number
            
            # Filter data relevant to this claim
            relevant_stmts = [
                s for s in all_statements
                if s.affected_claims and claim_num in s.affected_claims
            ]
            
            relevant_risks = [
                r for r in all_risks
                if r.affected_claims and claim_num in r.affected_claims
            ]
            
            # Build amendment history
            app_claim = next(
                (c for c in all_claims if c.mapped_final_claim_number == claim_num),
                None
            )
            amendment_text = "No amendment history available."
            if app_claim:
                versions = session.query(ClaimVersion).filter(
                    ClaimVersion.claim_id == app_claim.id,
                ).order_by(ClaimVersion.version_number).all()
                amendment_text = "\n".join(
                    f"V{v.version_number} ({v.date_of_change}): {v.change_summary or 'N/A'}"
                    for v in versions
                )
            
            prompt = VULNERABILITY_CARD_PROMPT.format(
                claim_number=claim_num,
                claim_text=fc.claim_text[:3000],
                prosecution_statements="\n".join(
                    f"- [{s.speaker}] {s.extracted_text[:300]}"
                    for s in relevant_stmts[:15]
                ) or "None recorded.",
                validity_risks="\n".join(
                    f"- [{r.severity}] {r.risk_type}: {r.description[:200]}"
                    for r in relevant_risks
                ) or "None recorded.",
                estoppel_events="\n".join(
                    f"- {s.extracted_text[:300]}"
                    for s in relevant_stmts
                    if s.relevance_category in ('Estoppel', 'Prior Art Distinction')
                )[:3000] or "None recorded.",
                amendment_history=amendment_text[:3000],
            )
            
            try:
                result = provider.complete_json(prompt, SYSTEM_PROMPT)
                
                card = ClaimVulnerabilityCard(
                    patent_id=patent_id,
                    claim_number=claim_num,
                    must_practice=result.get('must_practice'),
                    categorical_exclusions=result.get('categorical_exclusions'),
                    estoppel_bars=result.get('estoppel_bars'),
                    indefiniteness_targets=result.get('indefiniteness_targets'),
                    design_around_paths=result.get('design_around_paths'),
                    overall_vulnerability=result.get('overall_vulnerability', 'Medium'),
                )
                session.add(card)
                card_count += 1
                
            except Exception as e:
                logger.error(f"Vulnerability card failed for claim {claim_num}: {e}")
        
        logger.info(f"Vulnerability Cards: {card_count} cards generated")
    
    # =========================================================================
    # VERBOSE LOGGING
    # =========================================================================
    def _log_verbose(self, message: str, data: Any = None):
        if not self.verbose:
            return
        logger.info(f"[VERBOSE] {message}")
        if data is not None:
            data_str = json.dumps(data, indent=2, default=str) if isinstance(data, (dict, list)) else str(data)
            if len(data_str) > 3000:
                logger.info(f"[VERBOSE] Data (truncated):\n{data_str[:1500]}\n...\n{data_str[-1500:]}")
            else:
                logger.info(f"[VERBOSE] Data:\n{data_str}")
