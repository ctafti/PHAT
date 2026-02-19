"""
PHAT Verification Module
Verifies that AI-extracted quotes actually exist in the source text.

This module provides grounding verification to detect potential AI hallucinations
in legal document analysis, which is critical for patent prosecution accuracy.
"""

import logging
import re
from difflib import SequenceMatcher
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of quote verification"""
    VERIFIED = "verified"           # Exact or near-exact match found
    PARTIAL = "partial"             # Partial match found (possible paraphrase)
    UNVERIFIED = "unverified"       # No match found - potential hallucination
    SKIPPED = "skipped"             # Verification skipped (empty input, etc.)


@dataclass
class VerificationResult:
    """Result of a quote verification check"""
    status: VerificationStatus
    confidence: float               # 0.0 to 1.0
    matched_text: Optional[str]     # The actual text matched in source
    match_location: Optional[int]   # Character position in source
    details: str                    # Human-readable explanation
    
    @property
    def is_verified(self) -> bool:
        """Returns True if the quote was verified"""
        return self.status == VerificationStatus.VERIFIED
    
    @property
    def is_suspicious(self) -> bool:
        """Returns True if the quote might be hallucinated"""
        return self.status == VerificationStatus.UNVERIFIED


class ContentVerifier:
    """
    Verifies that AI-extracted content actually exists in the source document.
    
    This is crucial for legal document analysis where hallucinated quotes
    could lead to incorrect legal conclusions.
    """
    
    def __init__(
        self,
        exact_match_threshold: float = 1.0,
        fuzzy_match_threshold: float = 0.85,
        partial_match_threshold: float = 0.6,
        min_quote_length: int = 10,
        max_search_window: int = 50000
    ):
        """
        Initialize the content verifier.
        
        Args:
            exact_match_threshold: Confidence for exact matches (1.0)
            fuzzy_match_threshold: Minimum similarity for "verified" status
            partial_match_threshold: Minimum similarity for "partial" status
            min_quote_length: Minimum characters for meaningful verification
            max_search_window: Maximum source text length to search
        """
        self.exact_match_threshold = exact_match_threshold
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.partial_match_threshold = partial_match_threshold
        self.min_quote_length = min_quote_length
        self.max_search_window = max_search_window
    
    def verify_quote(
        self,
        source_text: str,
        extracted_quote: str,
        context_hint: Optional[str] = None
    ) -> VerificationResult:
        """
        Verify if the extracted_quote exists in source_text.
        
        Args:
            source_text: The original document text
            extracted_quote: The quote extracted by AI
            context_hint: Optional context to narrow search (e.g., "claim 1")
            
        Returns:
            VerificationResult with status, confidence, and details
        """
        # Handle edge cases
        if not extracted_quote or not source_text:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                confidence=0.0,
                matched_text=None,
                match_location=None,
                details="Empty input - verification skipped"
            )
        
        if len(extracted_quote.strip()) < self.min_quote_length:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                confidence=0.0,
                matched_text=None,
                match_location=None,
                details=f"Quote too short ({len(extracted_quote)} chars) for meaningful verification"
            )
        
        # Limit search window for performance
        search_text = source_text[:self.max_search_window]
        
        # Strategy 1: Exact match (fastest)
        result = self._check_exact_match(search_text, extracted_quote)
        if result.is_verified:
            return result
        
        # Strategy 2: Normalized match (handles whitespace/punctuation)
        result = self._check_normalized_match(search_text, extracted_quote)
        if result.is_verified:
            return result
        
        # Strategy 3: Fuzzy match (handles OCR errors, small variations)
        result = self._check_fuzzy_match(search_text, extracted_quote)
        if result.status != VerificationStatus.UNVERIFIED:
            return result
        
        # Strategy 4: Key phrase match (handles heavy paraphrasing)
        result = self._check_key_phrase_match(search_text, extracted_quote)
        if result.status != VerificationStatus.UNVERIFIED:
            return result
        
        # No match found
        return VerificationResult(
            status=VerificationStatus.UNVERIFIED,
            confidence=0.0,
            matched_text=None,
            match_location=None,
            details="Quote not found in source text - potential hallucination"
        )
    
    def _check_exact_match(self, source: str, quote: str) -> VerificationResult:
        """Check for exact substring match"""
        position = source.find(quote)
        if position != -1:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=self.exact_match_threshold,
                matched_text=quote,
                match_location=position,
                details="Exact match found"
            )
        return VerificationResult(
            status=VerificationStatus.UNVERIFIED,
            confidence=0.0,
            matched_text=None,
            match_location=None,
            details="No exact match"
        )
    
    def _check_normalized_match(self, source: str, quote: str) -> VerificationResult:
        """Check for match after normalizing whitespace and punctuation"""
        def normalize(s: str) -> str:
            # Remove extra whitespace, normalize to single spaces
            s = re.sub(r'\s+', ' ', s)
            # Remove common punctuation variations
            s = re.sub(r'[''`]', "'", s)
            s = re.sub(r'["""]', '"', s)
            # Lowercase for comparison
            return s.lower().strip()
        
        norm_source = normalize(source)
        norm_quote = normalize(quote)
        
        position = norm_source.find(norm_quote)
        if position != -1:
            # Find approximate position in original
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=0.95,
                matched_text=quote,
                match_location=position,
                details="Match found after normalization (whitespace/punctuation differences)"
            )
        return VerificationResult(
            status=VerificationStatus.UNVERIFIED,
            confidence=0.0,
            matched_text=None,
            match_location=None,
            details="No normalized match"
        )
    
    def _check_fuzzy_match(self, source: str, quote: str) -> VerificationResult:
        """
        Check for fuzzy match using sliding window comparison.
        
        This handles OCR errors and small textual variations.
        """
        quote_len = len(quote)
        best_ratio = 0.0
        best_match = None
        best_position = None
        
        # Slide a window of quote_len (with some tolerance) over the source
        window_sizes = [quote_len, int(quote_len * 0.9), int(quote_len * 1.1)]
        
        for window_size in window_sizes:
            if window_size < self.min_quote_length:
                continue
                
            step = max(1, window_size // 4)  # Overlap windows
            
            for i in range(0, len(source) - window_size + 1, step):
                window = source[i:i + window_size]
                
                # Quick pre-filter: skip if first/last chars don't match at all
                if quote and window:
                    if quote[0].lower() != window[0].lower() and quote[-1].lower() != window[-1].lower():
                        continue
                
                ratio = SequenceMatcher(None, quote.lower(), window.lower()).ratio()
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = window
                    best_position = i
                    
                    # Early exit if we find a very good match
                    if ratio >= self.fuzzy_match_threshold:
                        return VerificationResult(
                            status=VerificationStatus.VERIFIED,
                            confidence=ratio,
                            matched_text=best_match,
                            match_location=best_position,
                            details=f"Fuzzy match found (similarity: {ratio:.2%})"
                        )
        
        # Check best match against thresholds
        if best_ratio >= self.fuzzy_match_threshold:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=best_ratio,
                matched_text=best_match,
                match_location=best_position,
                details=f"Fuzzy match found (similarity: {best_ratio:.2%})"
            )
        elif best_ratio >= self.partial_match_threshold:
            return VerificationResult(
                status=VerificationStatus.PARTIAL,
                confidence=best_ratio,
                matched_text=best_match,
                match_location=best_position,
                details=f"Partial match found (similarity: {best_ratio:.2%}) - may be paraphrased"
            )
        
        return VerificationResult(
            status=VerificationStatus.UNVERIFIED,
            confidence=best_ratio,
            matched_text=best_match,
            match_location=best_position,
            details=f"Best fuzzy match too low (similarity: {best_ratio:.2%})"
        )
    
    def _check_key_phrase_match(self, source: str, quote: str) -> VerificationResult:
        """
        Check if key phrases from the quote appear in the source.
        
        This catches cases where the AI correctly identified content
        but significantly paraphrased it.
        """
        # Extract significant phrases (3+ words, excluding common words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
                      'as', 'that', 'which', 'this', 'these', 'those', 'and', 'or',
                      'but', 'if', 'then', 'than', 'so', 'such', 'said', 'claim',
                      'claims', 'wherein', 'comprising', 'including', 'having'}
        
        # Extract words from quote
        words = re.findall(r'\b[a-zA-Z]{3,}\b', quote.lower())
        significant_words = [w for w in words if w not in stop_words]
        
        if len(significant_words) < 3:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                confidence=0.0,
                matched_text=None,
                match_location=None,
                details="Not enough significant words for key phrase matching"
            )
        
        # Check how many significant words appear in source
        source_lower = source.lower()
        found_words = [w for w in significant_words if w in source_lower]
        match_ratio = len(found_words) / len(significant_words)
        
        if match_ratio >= 0.8:
            return VerificationResult(
                status=VerificationStatus.PARTIAL,
                confidence=match_ratio * 0.7,  # Lower confidence for key phrase match
                matched_text=None,
                match_location=None,
                details=f"Key phrases found ({len(found_words)}/{len(significant_words)} significant words) - likely paraphrased"
            )
        
        return VerificationResult(
            status=VerificationStatus.UNVERIFIED,
            confidence=match_ratio * 0.5,
            matched_text=None,
            match_location=None,
            details=f"Key phrase match too low ({len(found_words)}/{len(significant_words)} words found)"
        )
    
    def verify_multiple(
        self,
        source_text: str,
        quotes: List[str]
    ) -> List[VerificationResult]:
        """
        Verify multiple quotes against the same source text.
        
        Args:
            source_text: The original document text
            quotes: List of quotes to verify
            
        Returns:
            List of VerificationResults corresponding to each quote
        """
        return [self.verify_quote(source_text, q) for q in quotes]
    
    def get_verification_summary(
        self,
        results: List[VerificationResult]
    ) -> Dict[str, Any]:
        """
        Generate a summary of verification results.
        
        Args:
            results: List of verification results
            
        Returns:
            Summary dictionary with counts and statistics
        """
        if not results:
            return {
                'total': 0,
                'verified': 0,
                'partial': 0,
                'unverified': 0,
                'skipped': 0,
                'hallucination_risk': 'none',
                'average_confidence': 0.0
            }
        
        verified = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
        partial = sum(1 for r in results if r.status == VerificationStatus.PARTIAL)
        unverified = sum(1 for r in results if r.status == VerificationStatus.UNVERIFIED)
        skipped = sum(1 for r in results if r.status == VerificationStatus.SKIPPED)
        
        valid_results = [r for r in results if r.status != VerificationStatus.SKIPPED]
        avg_confidence = sum(r.confidence for r in valid_results) / len(valid_results) if valid_results else 0.0
        
        # Determine overall hallucination risk
        if unverified == 0:
            risk = 'low'
        elif unverified <= len(valid_results) * 0.1:
            risk = 'medium'
        else:
            risk = 'high'
        
        return {
            'total': len(results),
            'verified': verified,
            'partial': partial,
            'unverified': unverified,
            'skipped': skipped,
            'hallucination_risk': risk,
            'average_confidence': avg_confidence
        }


class StatementVerifier:
    """
    Specialized verifier for prosecution statements in patent analysis.
    
    Wraps ContentVerifier with patent-specific logic and integrates
    with the statement extraction workflow.
    """
    
    def __init__(self, verifier: Optional[ContentVerifier] = None):
        """
        Initialize the statement verifier.
        
        Args:
            verifier: ContentVerifier instance (creates default if None)
        """
        self.verifier = verifier or ContentVerifier()
        self.verification_log: List[Dict[str, Any]] = []
    
    def verify_statement(
        self,
        source_text: str,
        statement_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], VerificationResult]:
        """
        Verify a prosecution statement and annotate it with verification status.
        
        Args:
            source_text: The source document text
            statement_data: Dictionary containing statement info (must have 'extracted_text')
            
        Returns:
            Tuple of (annotated statement_data, VerificationResult)
        """
        quote = statement_data.get('extracted_text', '')
        
        result = self.verifier.verify_quote(source_text, quote)
        
        # Annotate the statement with verification info
        annotated = statement_data.copy()
        annotated['_verification'] = {
            'status': result.status.value,
            'confidence': result.confidence,
            'details': result.details
        }
        
        # Add warning to context if unverified
        if result.status == VerificationStatus.UNVERIFIED:
            existing_context = annotated.get('context', '') or ''
            annotated['context'] = f"[WARNING: Quote not verified in source text] {existing_context}".strip()
            logger.warning(f"Potential hallucination detected: '{quote[:50]}...'")
        
        # Log for audit trail
        self.verification_log.append({
            'quote': quote[:100] + '...' if len(quote) > 100 else quote,
            'status': result.status.value,
            'confidence': result.confidence
        })
        
        return annotated, result
    
    def verify_statements_batch(
        self,
        source_text: str,
        statements: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Verify a batch of statements and return annotated versions with summary.
        
        Args:
            source_text: The source document text
            statements: List of statement dictionaries
            
        Returns:
            Tuple of (annotated statements, verification summary)
        """
        annotated = []
        results = []
        
        for stmt in statements:
            ann_stmt, result = self.verify_statement(source_text, stmt)
            annotated.append(ann_stmt)
            results.append(result)
        
        summary = self.verifier.get_verification_summary(results)
        
        return annotated, summary
    
    def get_verification_log(self) -> List[Dict[str, Any]]:
        """Get the verification audit log"""
        return self.verification_log.copy()
    
    def clear_log(self):
        """Clear the verification audit log"""
        self.verification_log = []


# Factory function for easy instantiation from config
def create_content_verifier(config: Dict[str, Any]) -> ContentVerifier:
    """
    Create a content verifier from configuration.
    
    Args:
        config: Configuration dictionary with optional 'verification' section
        
    Returns:
        Configured ContentVerifier instance
    """
    ver_config = config.get('verification', {})
    return ContentVerifier(
        fuzzy_match_threshold=ver_config.get('fuzzy_threshold', 0.85),
        partial_match_threshold=ver_config.get('partial_threshold', 0.6),
        min_quote_length=ver_config.get('min_quote_length', 10)
    )


if __name__ == "__main__":
    # Test the verifier
    logging.basicConfig(level=logging.DEBUG)
    
    source_text = """
    Claims 1-5 are rejected under 35 U.S.C. 103 as being unpatentable over
    Smith (US 9,000,000) in view of Jones (US 8,500,000).
    
    Regarding claim 1, Smith teaches a method for processing data comprising
    receiving input from a sensor and generating an output signal based on
    the processed data.
    
    Jones teaches the use of machine learning algorithms for signal processing
    in similar technical environments.
    
    It would have been obvious to one of ordinary skill in the art to combine
    the teachings of Smith and Jones to achieve the claimed invention.
    """
    
    verifier = ContentVerifier()
    
    # Test exact match
    test1 = "receiving input from a sensor and generating an output signal"
    result1 = verifier.verify_quote(source_text, test1)
    print(f"Test 1 (exact): {result1.status.value} ({result1.confidence:.2%})")
    
    # Test with minor variation
    test2 = "receiving input from a sensor and generating  an output signal"  # extra space
    result2 = verifier.verify_quote(source_text, test2)
    print(f"Test 2 (variation): {result2.status.value} ({result2.confidence:.2%})")
    
    # Test hallucination
    test3 = "Smith teaches that artificial intelligence should always be used"
    result3 = verifier.verify_quote(source_text, test3)
    print(f"Test 3 (hallucination): {result3.status.value} ({result3.confidence:.2%})")
    
    # Test paraphrase
    test4 = "the examiner found claims obvious based on combining Smith and Jones references"
    result4 = verifier.verify_quote(source_text, test4)
    print(f"Test 4 (paraphrase): {result4.status.value} ({result4.confidence:.2%})")
