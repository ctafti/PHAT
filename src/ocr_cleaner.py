"""
PHAT OCR Cleaner Module
Fixes common OCR errors in patent documents with strictly scoped regex patterns.
Detects severely garbled text from corrupt PDF text layers and falls back to
Tesseract OCR on the page image when needed.

WARNING: Regex-based OCR correction is risky. This module uses conservative,
strictly scoped patterns to minimize the risk of data corruption.
"""

import re
import io
import logging
import statistics
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Optional imports for re-OCR functionality
try:
    import fitz  # PyMuPDF (legacy import)
    HAS_PYMUPDF = True
except Exception:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24.x preferred import
        HAS_PYMUPDF = True
    except Exception as e:
        HAS_PYMUPDF = False
        logger.debug(f"PyMuPDF not available - re-OCR fallback disabled: {e}")

try:
    from PIL import Image
    import pytesseract
    HAS_TESSERACT = True
except Exception as e:
    HAS_TESSERACT = False
    logger.debug(f"pytesseract/Pillow not available - re-OCR fallback disabled: {e}")


class GarbledTextDetector:
    """
    Detects garbled/corrupt text from bad OCR text layers in scanned PDFs.
    
    Uses multiple heuristics to identify text that was poorly encoded by the
    original OCR engine that created the PDF's invisible text layer:
    
    1. Special character ratio - garbled text has abnormally high punctuation/symbol density
    2. Consecutive punctuation runs - real text rarely has 3+ consecutive special chars
    3. Real word ratio - garbled text produces very few recognizable English words
    4. Character class entropy - garbled text mixes cases and symbols erratically
    
    Detection operates at line level, then aggregates to classify entire pages.
    """
    
    # Thresholds (tuned against the known garbled sample from US Patent 8,026,929)
    # The garbled line "nr-nr-oc,c,inn c,Hc,torn r-nnfin," has:
    #   - special_char_ratio ≈ 0.35 (vs ~0.10 for clean patent text)
    #   - consecutive punctuation runs of 3+ chars
    #   - near-zero real word ratio
    LINE_SPECIAL_CHAR_RATIO_THRESHOLD = 0.25   # Flag if >25% of chars are special
    LINE_CONSECUTIVE_PUNCT_THRESHOLD = 3        # Flag if 3+ consecutive special chars
    LINE_MIN_LENGTH = 20                         # Only evaluate lines with 20+ chars
    PAGE_GARBLED_LINE_RATIO_THRESHOLD = 0.10    # Re-OCR page if >10% of lines are garbled
    PAGE_MIN_LINES_FOR_DETECTION = 5            # Need at least 5 evaluable lines
    
    # Characters that count as "special" in the ratio calculation
    # Excludes spaces, periods, commas, hyphens (common in normal patent text)
    SPECIAL_CHARS = set('!@#$%^&*+=<>{}[]|\\~`:;\'\"()_')
    
    # Characters that count as punctuation runs (includes comma clusters and colons
    # which are hallmarks of the garbled text we saw)
    PUNCT_RUN_CHARS = set('!@#$%^&*+=<>{}[]|\\~`:;_!?')
    
    # Common English words for real-word ratio check (lowercase)
    # Focused on words that appear frequently in patent prosecution documents
    COMMON_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again',
        'and', 'but', 'or', 'nor', 'not', 'no', 'so', 'if', 'then', 'than',
        'that', 'this', 'these', 'those', 'which', 'what', 'where', 'when',
        'who', 'whom', 'how', 'each', 'every', 'all', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'only', 'own', 'same', 'also',
        # Patent-specific
        'claim', 'claims', 'method', 'system', 'apparatus', 'device',
        'comprising', 'configured', 'wherein', 'further', 'said', 'step',
        'receiving', 'generating', 'processing', 'determining', 'providing',
        'first', 'second', 'third', 'plurality', 'one', 'two', 'three',
        'data', 'signal', 'user', 'image', 'model', 'interface', 'computer',
        'based', 'according', 'including', 'corresponding', 'associated',
        'amended', 'canceled', 'rejected', 'allowed', 'pending', 'filed',
        'applicant', 'examiner', 'office', 'action', 'response', 'amendment',
        'rejection', 'prior', 'art', 'reference', 'cited', 'section',
        'currently', 'previously', 'originally', 'new', 'independent',
        'dependent', 'original', 'proposed', 'following', 'above', 'below',
    }
    
    @classmethod
    def is_line_garbled(cls, line: str) -> Tuple[bool, Dict[str, float]]:
        """
        Determine if a single line of text is garbled.
        
        Returns:
            Tuple of (is_garbled, metrics_dict)
        """
        stripped = line.strip()
        if len(stripped) < cls.LINE_MIN_LENGTH:
            return False, {'reason': 'too_short'}
        
        metrics = {}
        flags = 0
        
        # --- Heuristic 1: Special character ratio ---
        special_count = sum(1 for c in stripped if c in cls.SPECIAL_CHARS)
        total_nonspace = sum(1 for c in stripped if not c.isspace())
        if total_nonspace > 0:
            special_ratio = special_count / total_nonspace
            metrics['special_char_ratio'] = round(special_ratio, 3)
            if special_ratio > cls.LINE_SPECIAL_CHAR_RATIO_THRESHOLD:
                flags += 1
        
        # --- Heuristic 2: Consecutive punctuation/symbol runs ---
        # Look for runs of 3+ characters from the punct set (including commas
        # and colons when in runs, which are hallmarks of garbled text)
        # The pattern: 3+ chars that are any of the special chars OR commas/colons
        # when mixed with other specials
        max_run = 0
        current_run = 0
        for c in stripped:
            if c in cls.PUNCT_RUN_CHARS or (c in ',.:' and current_run > 0):
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        metrics['max_punct_run'] = max_run
        if max_run >= cls.LINE_CONSECUTIVE_PUNCT_THRESHOLD:
            flags += 1
        
        # --- Heuristic 3: Real word ratio ---
        # Split on whitespace and non-alpha, check what fraction are real words
        words = re.findall(r'[a-zA-Z]{2,}', stripped)
        if len(words) >= 3:
            real_count = sum(1 for w in words if w.lower() in cls.COMMON_WORDS)
            word_ratio = real_count / len(words)
            metrics['real_word_ratio'] = round(word_ratio, 3)
            metrics['total_words'] = len(words)
            # Very low real word ratio is suspicious
            if word_ratio < 0.10:
                flags += 1
        
        # --- Heuristic 4: Fragmented short "words" ---
        # Garbled text like "c,Hc,torn r-nnfin," produces many 1-2 char tokens
        tokens = stripped.split()
        if len(tokens) >= 3:
            short_tokens = sum(1 for t in tokens if len(t) <= 2)
            short_ratio = short_tokens / len(tokens)
            metrics['short_token_ratio'] = round(short_ratio, 3)
            if short_ratio > 0.40:
                flags += 1
        
        # --- Decision: garbled if 2+ heuristics flagged ---
        is_garbled = flags >= 2
        metrics['flags'] = flags
        
        return is_garbled, metrics
    
    @classmethod
    def analyze_page(cls, page_text: str) -> Dict[str, Any]:
        """
        Analyze a full page of text for garbled content.
        
        Returns:
            Dictionary with:
                - is_garbled: bool - whether the page needs re-OCR
                - garbled_line_count: int
                - total_evaluated_lines: int
                - garbled_ratio: float
                - garbled_lines: list of (line_number, line_text, metrics)
        """
        lines = page_text.split('\n')
        garbled_lines = []
        evaluated_count = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) < cls.LINE_MIN_LENGTH:
                continue
            
            evaluated_count += 1
            is_garbled, metrics = cls.is_line_garbled(line)
            
            if is_garbled:
                garbled_lines.append({
                    'line_number': i + 1,
                    'text': stripped[:200],
                    'metrics': metrics
                })
        
        garbled_count = len(garbled_lines)
        
        # Calculate garbled ratio
        if evaluated_count < cls.PAGE_MIN_LINES_FOR_DETECTION:
            garbled_ratio = 0.0
            is_page_garbled = False
        else:
            garbled_ratio = garbled_count / evaluated_count
            is_page_garbled = garbled_ratio > cls.PAGE_GARBLED_LINE_RATIO_THRESHOLD
        
        # Also flag if there are ANY severely garbled lines (3+ flags)
        # Even one severely garbled line in a claim section is worth re-OCR
        has_severe_garble = any(
            gl['metrics'].get('flags', 0) >= 3 for gl in garbled_lines
        )
        
        return {
            'is_garbled': is_page_garbled or has_severe_garble,
            'garbled_line_count': garbled_count,
            'total_evaluated_lines': evaluated_count,
            'garbled_ratio': round(garbled_ratio, 3),
            'has_severe_garble': has_severe_garble,
            'garbled_lines': garbled_lines
        }


class PageReOCR:
    """
    Re-OCRs a PDF page image using Tesseract when the embedded text layer is corrupt.
    
    Workflow:
    1. Render the PDF page to a high-DPI pixmap via PyMuPDF
    2. Convert to PIL Image
    3. Run Tesseract OCR
    4. Return the fresh text
    
    This mimics what your PDF viewer does when it detects a bad text layer —
    it falls back to its own OCR on the visible image.
    """
    
    # Tesseract config for patent documents
    # PSM 6 = Assume a single uniform block of text (good for full pages)
    # OEM 3 = Default, based on what is available (LSTM + legacy)
    DEFAULT_TESSERACT_CONFIG = '--psm 6 --oem 3'
    
    # DPI for rendering - higher = better OCR but slower
    # 300 DPI is standard for document OCR
    DEFAULT_DPI = 300
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if re-OCR dependencies are available."""
        return HAS_PYMUPDF and HAS_TESSERACT
    
    @classmethod
    def reocr_page(
        cls,
        doc: 'fitz.Document',
        page_index: int,
        dpi: int = DEFAULT_DPI,
        tesseract_config: str = DEFAULT_TESSERACT_CONFIG,
    ) -> Optional[str]:
        """
        Re-OCR a single page from a PDF document.
        
        Args:
            doc: PyMuPDF Document object (already opened)
            page_index: 0-based page index
            dpi: Resolution for rendering (default 300)
            tesseract_config: Tesseract CLI config string
            
        Returns:
            OCR'd text string, or None if re-OCR failed
        """
        if not cls.is_available():
            logger.warning("Re-OCR requested but dependencies not available")
            return None
        
        try:
            page = doc[page_index]
            
            # Check that this page actually has an image (is scanned)
            images = page.get_images()
            if not images:
                logger.debug(f"Page {page_index + 1}: No images found, skipping re-OCR")
                return None
            
            # Render page to pixmap at target DPI
            # Default PDF resolution is 72 DPI, so scale factor = target_dpi / 72
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=mat)
            
            # Convert pixmap to PIL Image
            img_data = pixmap.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Run Tesseract
            ocr_text = pytesseract.image_to_string(
                pil_image,
                config=tesseract_config
            )
            
            logger.info(
                f"Re-OCR page {page_index + 1}: "
                f"{len(ocr_text)} chars extracted via Tesseract "
                f"(image {pixmap.width}x{pixmap.height} @ {dpi} DPI)"
            )
            
            return ocr_text
            
        except Exception as e:
            logger.error(f"Re-OCR failed for page {page_index + 1}: {e}")
            return None
    
    @classmethod
    def reocr_pages(
        cls,
        doc: 'fitz.Document',
        page_indices: List[int],
        dpi: int = DEFAULT_DPI,
        tesseract_config: str = DEFAULT_TESSERACT_CONFIG,
    ) -> Dict[int, Optional[str]]:
        """
        Re-OCR multiple pages.
        
        Args:
            doc: PyMuPDF Document object
            page_indices: List of 0-based page indices to re-OCR
            dpi: Resolution for rendering
            tesseract_config: Tesseract CLI config string
            
        Returns:
            Dict mapping page_index -> OCR'd text (or None if failed)
        """
        results = {}
        for idx in page_indices:
            results[idx] = cls.reocr_page(doc, idx, dpi, tesseract_config)
        return results


class OCRCleaner:
    """
    Cleans common OCR errors in patent prosecution documents.
    
    Design Principles:
    1. Only fix high-confidence patterns (claim headers, common legal terms)
    2. Scope patterns strictly (start of line, specific contexts)
    3. Never apply global replacements to body text
    4. Log all corrections for auditability
    5. Detect severely garbled text and fall back to Tesseract re-OCR
    """
    
    def __init__(
        self,
        enabled: bool = True,
        log_corrections: bool = True,
        reocr_enabled: bool = True,
        reocr_dpi: int = 300,
    ):
        """
        Initialize the OCR cleaner.
        
        Args:
            enabled: Whether cleaning is active
            log_corrections: Whether to log each correction made
            reocr_enabled: Whether garbled text detection + Tesseract fallback is active
            reocr_dpi: DPI for Tesseract page rendering (default 300)
        """
        self.enabled = enabled
        self.log_corrections = log_corrections
        self.reocr_enabled = reocr_enabled and PageReOCR.is_available()
        self.reocr_dpi = reocr_dpi
        self.corrections_made: List[Dict[str, Any]] = []
        self.reocr_pages_replaced: List[Dict[str, Any]] = []
        
        if reocr_enabled and not PageReOCR.is_available():
            missing = []
            if not HAS_PYMUPDF:
                missing.append("PyMuPDF (pip install PyMuPDF)")
            if not HAS_TESSERACT:
                missing.append("pytesseract/Pillow (pip install pytesseract Pillow) + Tesseract binary")
            logger.warning(
                f"Re-OCR was requested but dependencies are missing: {', '.join(missing)}. "
                f"Falling back to regex-only cleaning."
            )
        
        # Define strictly scoped patterns
        self._patterns = self._build_patterns()
    
    def _build_patterns(self) -> List[Tuple[re.Pattern, str, str]]:
        """
        Build the list of strictly scoped OCR correction patterns.
        
        Each pattern is designed to be conservative and context-specific
        to avoid corrupting valid text.
        """
        patterns = []
        
        # =================================================================
        # CLAIM HEADER CORRECTIONS
        # These patterns only match claim numbers at line start
        # =================================================================
        
        # "Claim l" -> "Claim 1" (lowercase L to numeral 1)
        patterns.append((
            re.compile(r'^(\s*Claim\s+)l(\s*[.\s])', re.MULTILINE | re.IGNORECASE),
            r'\g<1>1\g<2>',
            "Claim l -> Claim 1 (header)"
        ))
        
        # "Claim I" -> "Claim 1" (capital I to numeral 1) - only standalone
        patterns.append((
            re.compile(r'^(\s*Claim\s+)I(\.\s)', re.MULTILINE | re.IGNORECASE),
            r'\g<1>1\g<2>',
            "Claim I -> Claim 1 (header, before period)"
        ))
        
        # Fix claim number with lowercase l in middle: "Claim l2" -> "Claim 12"
        patterns.append((
            re.compile(r'^(\s*Claim\s+)l(\d+)', re.MULTILINE | re.IGNORECASE),
            r'\g<1>1\g<2>',
            "Claim lX -> Claim 1X (tens digit)"
        ))
        
        # Fix "Claims l-5" -> "Claims 1-5"
        patterns.append((
            re.compile(r'^(\s*Claims?\s+)l(\s*[-–—]\s*\d+)', re.MULTILINE | re.IGNORECASE),
            r'\g<1>1\g<2>',
            "Claims l-X -> Claims 1-X (range)"
        ))
        
        # =================================================================
        # NUMBERED LIST CORRECTIONS (at line start only)
        # =================================================================
        
        # "l." at line start -> "1." (numbered list item)
        patterns.append((
            re.compile(r'^(\s*)l\.(\s+[A-Z])', re.MULTILINE),
            r'\g<1>1.\g<2>',
            "l. -> 1. (list start)"
        ))
        
        # "l)" at line start -> "1)" (numbered list item, paren style)
        patterns.append((
            re.compile(r'^(\s*)l\)(\s+[A-Z])', re.MULTILINE),
            r'\g<1>1)\g<2>',
            "l) -> 1) (list start)"
        ))
        
        # =================================================================
        # SECTION HEADER CORRECTIONS
        # =================================================================
        
        # "35 U.S.C. § l02" -> "35 U.S.C. § 102"
        patterns.append((
            re.compile(r'(35\s*U\.?S\.?C\.?\s*§?\s*)l(0[0-9])', re.IGNORECASE),
            r'\g<1>1\g<2>',
            "35 USC l0X -> 35 USC 10X"
        ))
        
        # "35 U.S.C. l12" -> "35 U.S.C. 112"
        patterns.append((
            re.compile(r'(35\s*U\.?S\.?C\.?\s*§?\s*)l(12)', re.IGNORECASE),
            r'\g<1>1\g<2>',
            "35 USC l12 -> 35 USC 112"
        ))
        
        # =================================================================
        # COMMON OCR LIGATURE/CHARACTER ISSUES
        # =================================================================
        
        patterns.append((
            re.compile(r'(speci)ÿ(cation)', re.IGNORECASE),
            r'\g<1>fi\g<2>',
            "speciŸcation -> specification"
        ))
        
        patterns.append((
            re.compile(r'(identi)ÿ(ed|es|cation)', re.IGNORECASE),
            r'\g<1>fi\g<2>',
            "identiŸed -> identified"
        ))
        
        # "rn" sometimes OCRs as "m" - only in specific headers
        patterns.append((
            re.compile(r'^(\s*GOVE)(M)(MENT)', re.MULTILINE),
            r'\g<1>RN\g<3>',
            "GOVEMMENT -> GOVERNMENT (header)"
        ))
        
        # =================================================================
        # ZERO/LETTER O CONFUSION IN SPECIFIC CONTEXTS
        # =================================================================
        
        # -----------------------------------------------------------------
        # DIMENSION NOTATION: "20 image" -> "2D image" (Bug Fix B, v2.2)
        # OCR frequently renders "2D" as "20", "3D" as "30", etc. because
        # the letter D and digit 0 are visually similar.
        # Universal fix: handles 0D through 4D in context of spatial/graphics terms.
        # -----------------------------------------------------------------
        DIMENSION_CONTEXTS = (
            'image|model|graphic|object|space|rendering|display|view|animation'
            '|coordinate|mapping|map|representation|projection|data|scan|scanner'
            '|printing|printer|print|array|matrix|vector|scene|environment|world'
            '|surface|texture|mesh|geometry|visualization|content|video|camera'
        )
        
        # "[0-4]0 <spatial_term>" -> "[0-4]D <spatial_term>"
        patterns.append((
            re.compile(
                rf'\b([0-4])0\s+({DIMENSION_CONTEXTS})',
                re.IGNORECASE
            ),
            r'\g<1>D \g<2>',
            "N0 <spatial_term> -> ND <spatial_term> (OCR dimension fix)"
        ))
        
        # Also catch hyphenated form: "20-image" -> "2D-image"
        patterns.append((
            re.compile(
                rf'\b([0-4])0-({DIMENSION_CONTEXTS})',
                re.IGNORECASE
            ),
            r'\g<1>D-\g<2>',
            "N0-<spatial_term> -> ND-<spatial_term> (OCR dimension fix, hyphenated)"
        ))
        
        # Patent numbers: "US l0,000,000" -> "US 10,000,000"
        patterns.append((
            re.compile(r'(US\s*)l([\d,]{7,})', re.IGNORECASE),
            r'\g<1>1\g<2>',
            "US lX,XXX,XXX -> US 1X,XXX,XXX"
        ))
        
        # Application numbers: "l6/123,456" -> "16/123,456"
        patterns.append((
            re.compile(r'^(\s*)l(\d/[\d,]+)', re.MULTILINE),
            r'\g<1>1\g<2>',
            "lX/XXX,XXX -> 1X/XXX,XXX (app number)"
        ))
        
        return patterns
    
    def clean(self, text: str) -> str:
        """
        Apply OCR corrections to the given text.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text with OCR corrections applied
        """
        if not self.enabled or not text:
            return text
        
        self.corrections_made = []
        
        for pattern, replacement, description in self._patterns:
            matches = list(pattern.finditer(text))
            
            if matches and self.log_corrections:
                for match in matches:
                    self.corrections_made.append({
                        'pattern': description,
                        'original': match.group(0),
                        'position': match.start(),
                        'context': text[max(0, match.start()-20):min(len(text), match.end()+20)]
                    })
            
            text = pattern.sub(replacement, text)
        
        if self.corrections_made:
            logger.info(f"OCR Cleaner: Made {len(self.corrections_made)} corrections")
            if self.log_corrections:
                for correction in self.corrections_made:
                    logger.debug(f"  OCR Fix [{correction['pattern']}]: "
                               f"'{correction['original']}' in context: ...{correction['context']}...")
        
        return text
    
    def clean_pages(self, pages: List[str]) -> List[str]:
        """
        Apply OCR corrections to a list of page texts.
        
        Args:
            pages: List of page text strings
            
        Returns:
            List of cleaned page texts
        """
        return [self.clean(page) for page in pages]
    
    def clean_pages_with_reocr(
        self,
        pages: List[str],
        pdf_path: str,
        skip_reocr_indices: Optional[set] = None,
    ) -> List[str]:
        """
        Apply OCR corrections with garbled text detection and Tesseract fallback.
        
        This is the primary entry point when a PDF path is available. For each page:
        1. Run garbled text detection heuristics
        2. If garbled AND page has an image layer → re-OCR via Tesseract
        3. Apply regex-based cleaning to the result (original or re-OCR'd)
        
        Args:
            pages: List of page text strings (from PyMuPDF text extraction)
            pdf_path: Path to the source PDF file (needed for image extraction)
            skip_reocr_indices: Optional set of page indices to exclude from re-OCR
                (Refinement B: pages belonging to low-value sections like Examiner
                Search or Citation List are skipped to save resources)
            
        Returns:
            List of cleaned page texts
        """
        self.reocr_pages_replaced = []
        
        if not self.enabled:
            return pages
        
        if skip_reocr_indices is None:
            skip_reocr_indices = set()
        
        # Phase 1: Detect garbled pages
        garbled_indices = []
        if self.reocr_enabled:
            for i, page_text in enumerate(pages):
                # Refinement B: skip re-OCR detection for low-value pages
                if i in skip_reocr_indices:
                    logger.debug(f"Page {i + 1}: Skipping garbled detection (low-value section)")
                    continue
                analysis = GarbledTextDetector.analyze_page(page_text)
                if analysis['is_garbled']:
                    garbled_indices.append(i)
                    logger.warning(
                        f"Page {i + 1}: Garbled text detected "
                        f"({analysis['garbled_line_count']}/{analysis['total_evaluated_lines']} lines, "
                        f"ratio={analysis['garbled_ratio']}, severe={analysis['has_severe_garble']})"
                    )
                    if self.log_corrections:
                        for gl in analysis['garbled_lines'][:3]:  # Log first 3
                            logger.debug(
                                f"  Garbled line {gl['line_number']}: "
                                f"{gl['text'][:100]}... "
                                f"(metrics: {gl['metrics']})"
                            )
        
        # Phase 2: Re-OCR garbled pages
        if garbled_indices and self.reocr_enabled:
            logger.info(
                f"Re-OCR: {len(garbled_indices)} garbled page(s) detected, "
                f"attempting Tesseract fallback for pages: "
                f"{[i + 1 for i in garbled_indices]}"
            )
            
            try:
                doc = fitz.open(pdf_path)
                reocr_results = PageReOCR.reocr_pages(
                    doc, garbled_indices, dpi=self.reocr_dpi
                )
                doc.close()
                
                for idx, new_text in reocr_results.items():
                    if new_text and len(new_text.strip()) > 0:
                        old_text = pages[idx]
                        pages[idx] = new_text
                        
                        # Validate that re-OCR actually improved things
                        old_analysis = GarbledTextDetector.analyze_page(old_text)
                        new_analysis = GarbledTextDetector.analyze_page(new_text)
                        
                        if new_analysis['garbled_line_count'] >= old_analysis['garbled_line_count']:
                            # Re-OCR didn't help or made things worse — revert
                            pages[idx] = old_text
                            logger.warning(
                                f"Page {idx + 1}: Re-OCR did not improve text quality "
                                f"(old garbled: {old_analysis['garbled_line_count']}, "
                                f"new garbled: {new_analysis['garbled_line_count']}). "
                                f"Keeping original text layer."
                            )
                        else:
                            self.reocr_pages_replaced.append({
                                'page_number': idx + 1,
                                'old_garbled_lines': old_analysis['garbled_line_count'],
                                'new_garbled_lines': new_analysis['garbled_line_count'],
                                'old_char_count': len(old_text),
                                'new_char_count': len(new_text),
                            })
                            logger.info(
                                f"Page {idx + 1}: Re-OCR successful — "
                                f"garbled lines {old_analysis['garbled_line_count']} → "
                                f"{new_analysis['garbled_line_count']}, "
                                f"chars {len(old_text)} → {len(new_text)}"
                            )
                    else:
                        logger.warning(
                            f"Page {idx + 1}: Re-OCR returned empty text, keeping original"
                        )
                        
            except Exception as e:
                logger.error(f"Re-OCR phase failed: {e}. Falling back to regex-only cleaning.")
        
        # Phase 3: Apply regex-based cleaning to all pages
        cleaned_pages = self.clean_pages(pages)
        
        return cleaned_pages
    
    def get_corrections_report(self) -> Dict[str, Any]:
        """
        Get a summary report of corrections made in the last clean() call.
        
        Returns:
            Dictionary with correction statistics and details
        """
        report = {}
        
        # Regex corrections
        if not self.corrections_made:
            report['total_corrections'] = 0
            report['corrections'] = []
        else:
            by_pattern = {}
            for c in self.corrections_made:
                pattern = c['pattern']
                if pattern not in by_pattern:
                    by_pattern[pattern] = []
                by_pattern[pattern].append(c)
            
            report['total_corrections'] = len(self.corrections_made)
            report['by_pattern'] = {k: len(v) for k, v in by_pattern.items()}
            report['corrections'] = self.corrections_made
        
        # Re-OCR results
        report['reocr_pages_replaced'] = len(self.reocr_pages_replaced)
        report['reocr_details'] = self.reocr_pages_replaced
        
        return report


class ClaimNumberNormalizer:
    """
    Specialized normalizer for claim references in patent documents.
    
    This handles the common case where claim numbers are corrupted by OCR
    but we need to reliably extract claim dependencies and references.
    """
    
    CLAIM_REF_PATTERN = re.compile(
        r'(?:claim|claims)\s+([lI1]?\d+(?:\s*[-–—,]\s*[lI1]?\d+)*)',
        re.IGNORECASE
    )
    
    @classmethod
    def normalize_claim_reference(cls, text: str) -> str:
        """
        Normalize claim number references in text.
        """
        def fix_claim_numbers(match):
            full_match = match.group(0)
            numbers_part = match.group(1)
            
            fixed_numbers = re.sub(r'([lI])(\d)', r'1\2', numbers_part)
            fixed_numbers = re.sub(r'^[lI]$', '1', fixed_numbers)
            fixed_numbers = re.sub(r'([,\-–—]\s*)[lI](\d)', r'\g<1>1\2', fixed_numbers)
            
            return full_match.replace(numbers_part, fixed_numbers)
        
        return cls.CLAIM_REF_PATTERN.sub(fix_claim_numbers, text)
    
    @classmethod
    def extract_claim_numbers(cls, text: str) -> List[int]:
        """
        Extract claim numbers from text, handling OCR errors.
        """
        normalized = cls.normalize_claim_reference(text)
        
        numbers = []
        for match in cls.CLAIM_REF_PATTERN.finditer(normalized):
            numbers_str = match.group(1)
            
            range_match = re.search(r'(\d+)\s*[-–—]\s*(\d+)', numbers_str)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                numbers.extend(range(start, end + 1))
            else:
                for num_str in re.findall(r'\d+', numbers_str):
                    numbers.append(int(num_str))
        
        return sorted(set(numbers))


# Factory function for easy instantiation from config
def create_ocr_cleaner(config: Dict[str, Any]) -> OCRCleaner:
    """
    Create an OCR cleaner from configuration.
    
    Args:
        config: Configuration dictionary with optional 'ocr' section
        
    Returns:
        Configured OCRCleaner instance
    """
    ocr_config = config.get('ocr', {})
    return OCRCleaner(
        enabled=ocr_config.get('enabled', True),
        log_corrections=ocr_config.get('log_corrections', True),
        reocr_enabled=ocr_config.get('reocr_enabled', True),
        reocr_dpi=ocr_config.get('reocr_dpi', 300),
    )


if __name__ == "__main__":
    # Test the OCR cleaner
    logging.basicConfig(level=logging.DEBUG)
    
    # Test 1: Regex-based cleaning
    print("=" * 60)
    print("TEST 1: Regex-based OCR corrections")
    print("=" * 60)
    
    test_text = """
    Claim l. A method comprising:
    Claim l2. The method of claim l, further comprising...
    Claims l-5 are rejected under 35 U.S.C. § l03.
    
    l. First step of the method
    2. Second step
    
    The specification describes...
    US l0,456,789 teaches...
    Application l6/123,456 claims...
    """
    
    cleaner = OCRCleaner()
    cleaned = cleaner.clean(test_text)
    
    print("Original:")
    print(test_text)
    print("\nCleaned:")
    print(cleaned)
    print("\nCorrections Report:")
    report = cleaner.get_corrections_report()
    print(f"  Total: {report['total_corrections']}")
    for pattern, count in report.get('by_pattern', {}).items():
        print(f"  {pattern}: {count}")
    
    # Test 2: Garbled text detection
    print("\n" + "=" * 60)
    print("TEST 2: Garbled text detection")
    print("=" * 60)
    
    clean_text = """Application No. 11/768,732 Attorney Docket No. 028080-0277
AMENDMENTS TO THE CLAIMS
Please amend the claims as follows:
1. (Currently amended) A system for generating pose data for posing a 2D
image of a scene aligned within the environment of the scene as rendered by a 3D
model of the environment, the system comprising:
a user interface configured to receive a digital representation of the 2D image;
a user interface configured to allow the user to specify an approximate location of
the scene on a two-dimensional geographic map;"""
    
    garbled_text = (
        "Application No. 11/768,732 Attorney Docket No. 028080-0277\n"
        "AMENDMENTS TO THE CLAIMS\n"
        "Please amend the claims as follows:\n"
        "1. (Currently amended) A system for generating pose data for posing a 2D\n"
        "image of a scene aligned within the environment of the scene as rendered by a 3D\n"
        "model of the environment, the system comprising:\n"
        "a user interface configured to receive a digital representation of the 2D image;\n"
        "a nr-nr-oc,c,inn c,Hc,torn r-nnfin, ,rorl tn nonor3+0 tho nnco rl!:!f!:I h!:lc<=>rl nn th,::, f,::,!:lfi 1r,::,c\n"
        "JJlVV\\,,.,...;,..;,111~ ._.,,~l.'-'111 V\"-Jllll~Ul'-'\\.A l,\\,,,J ~'-'lt'-'1 L'-' 1,11'-'\n"
        "mapped between the 2D image and the rendering of the 3D model by the user."
    )
    
    print("\nClean page analysis:")
    clean_analysis = GarbledTextDetector.analyze_page(clean_text)
    print(f"  is_garbled: {clean_analysis['is_garbled']}")
    print(f"  garbled lines: {clean_analysis['garbled_line_count']}/{clean_analysis['total_evaluated_lines']}")
    
    print("\nGarbled page analysis:")
    garbled_analysis = GarbledTextDetector.analyze_page(garbled_text)
    print(f"  is_garbled: {garbled_analysis['is_garbled']}")
    print(f"  garbled lines: {garbled_analysis['garbled_line_count']}/{garbled_analysis['total_evaluated_lines']}")
    print(f"  severe: {garbled_analysis['has_severe_garble']}")
    for gl in garbled_analysis['garbled_lines']:
        print(f"  Line {gl['line_number']}: {gl['text'][:80]}...")
        print(f"    metrics: {gl['metrics']}")
    
    # Test 3: Line-level detection on known garbled lines
    print("\n" + "=" * 60)
    print("TEST 3: Line-level garbled detection")
    print("=" * 60)
    
    test_lines = [
        ("CLEAN", "a user interface configured to receive a digital representation of the 2D image;"),
        ("CLEAN", "Please amend the claims as follows:"),
        ("GARBLED", "a nr-nr-oc,c,inn c,Hc,torn r-nnfin, ,rorl tn nonor3+0 tho nnco rl!:!f!:I h!:lc<=>rl nn th,::, f,::,!:lfi 1r,::,c"),
        ("GARBLED", 'JJlVV\\,,.,...;,..;,111~ ._.,,~l.\'-\'111 V"-Jllll~Ul\'-\'\\.A l,\\,,,J ~\'-\'lt\'-\'1 L\'-\' 1,11\'-\'"'),
    ]
    
    for expected, line in test_lines:
        is_garbled, metrics = GarbledTextDetector.is_line_garbled(line)
        status = "GARBLED" if is_garbled else "CLEAN"
        match = "✓" if status == expected else "✗ MISMATCH"
        print(f"  [{match}] Expected={expected}, Got={status}")
        print(f"    Line: {line[:80]}...")
        print(f"    Metrics: {metrics}")
    
    print("\n" + "=" * 60)
    print("TEST 4: Re-OCR availability")
    print("=" * 60)
    print(f"  PyMuPDF available: {HAS_PYMUPDF}")
    print(f"  Tesseract available: {HAS_TESSERACT}")
    print(f"  Re-OCR ready: {PageReOCR.is_available()}")
