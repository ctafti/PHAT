"""
PHAT Chunking Module v2.0
Handles splitting large document SECTIONS and merging AI results.

IMPORTANT: This is NOT for splitting the PDF into documents.
PDF bookmarks handle that (see pdf_processor.py).

This module is ONLY used when an individual document section
(e.g., a very long Office Action) exceeds the AI context window (~28K chars).
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Metadata about a text chunk within a section"""
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    total_chunks: int
    is_first: bool
    is_last: bool


class DocumentChunker:
    """
    Splits large document sections into overlapping chunks for AI processing.
    
    Only used when a single bookmark section exceeds max_chunk_chars.
    """
    
    def __init__(
        self,
        max_chunk_chars: int = 28000,
        overlap_chars: int = 1500,
        min_chunk_chars: int = 5000,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars
    
    def needs_chunking(self, text: str) -> bool:
        """Check if text needs to be chunked"""
        return len(text) > self.max_chunk_chars
    
    def chunk_text(self, text: str) -> List[ChunkInfo]:
        """Split text into overlapping chunks with smart break points"""
        if not self.needs_chunking(text):
            return [ChunkInfo(
                text=text, start_char=0, end_char=len(text),
                chunk_index=0, total_chunks=1, is_first=True, is_last=True,
            )]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.max_chunk_chars
            remaining = len(text) - start
            
            # If remaining fits in one more chunk, take it all
            if remaining <= self.max_chunk_chars + self.min_chunk_chars:
                end = len(text)
            else:
                end = self._find_break_point(text, start, end)
            
            chunks.append(ChunkInfo(
                text=text[start:end],
                start_char=start,
                end_char=end,
                chunk_index=chunk_index,
                total_chunks=-1,  # set below
                is_first=(chunk_index == 0),
                is_last=(end >= len(text)),
            ))
            
            if end >= len(text):
                break
            start = end - self.overlap_chars
            chunk_index += 1
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        logger.info(f"Split {len(text)} chars into {len(chunks)} chunks")
        return chunks
    
    def _find_break_point(self, text: str, start: int, ideal_end: int) -> int:
        """Find a good break point near ideal_end (paragraph > header > sentence > newline)"""
        search_start = max(start + self.min_chunk_chars, ideal_end - 2000)
        search_text = text[search_start:ideal_end]
        
        # Try paragraph break
        para_match = None
        for match in re.finditer(r'\n\s*\n', search_text):
            para_match = match
        if para_match:
            return search_start + para_match.end()
        
        # Try section headers
        for pattern in [r'\n\s*(?:CLAIM|REMARKS|ARGUMENTS?|REJECTION|RESPONSE)',
                        r'\n\s*\d+\.\s+', r'\n\s*[A-Z][A-Z\s]{5,}:']:
            matches = list(re.finditer(pattern, search_text, re.IGNORECASE))
            if matches:
                return search_start + matches[-1].start()
        
        # Try sentence ending
        sent_match = None
        for match in re.finditer(r'[.!?]\s+', search_text):
            sent_match = match
        if sent_match:
            return search_start + sent_match.end()
        
        # Try any newline
        nl_match = None
        for match in re.finditer(r'\n', search_text):
            nl_match = match
        if nl_match:
            return search_start + nl_match.end()
        
        return ideal_end


class ResultMerger:
    """
    Merges and deduplicates AI results from multiple chunks.
    Each result type has a specialized merge strategy.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    # =========================================================================
    # CLAIMS
    # =========================================================================
    
    def merge_claims(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deduplicate by claim number, keep first occurrence"""
        seen = set()
        merged = []
        for result in results:
            for claim in result.get('claims', []):
                num = claim.get('number')
                if num and num not in seen:
                    seen.add(num)
                    merged.append(claim)
        merged.sort(key=lambda c: c.get('number', 0))
        return {'claims': merged}

    # =========================================================================
    # REJECTIONS
    # =========================================================================
    
    def merge_rejections(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deduplicate by (statutory_basis, affected_claims, prior_art)"""
        is_final = any(r.get('is_final_action', False) for r in results)
        
        unique_rejections = []
        rejection_keys = set()
        for result in results:
            for rej in result.get('rejections', []):
                claims = tuple(sorted(rej.get('affected_claims', [])))
                basis = rej.get('statutory_basis', '')
                art_refs = []
                for art in rej.get('prior_art', []):
                    if isinstance(art, dict):
                        art_refs.append(art.get('reference', ''))
                    else:
                        art_refs.append(str(art))
                key = (claims, basis, tuple(sorted(art_refs)))
                if key not in rejection_keys:
                    rejection_keys.add(key)
                    unique_rejections.append(rej)
        
        unique_objections = []
        objection_keys = set()
        for result in results:
            for obj in result.get('objections', []):
                claims = tuple(sorted(obj.get('affected_claims', [])))
                key = (claims, obj.get('type', ''), obj.get('reason', '')[:50])
                if key not in objection_keys:
                    objection_keys.add(key)
                    unique_objections.append(obj)
        
        return {
            'is_final_action': is_final,
            'rejections': unique_rejections,
            'objections': unique_objections,
        }

    # =========================================================================
    # STATEMENTS / ARGUMENTS
    # =========================================================================
    
    def merge_statements(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deduplicate by text similarity"""
        unique = []
        for result in results:
            for stmt in result.get('statements', []):
                text = stmt.get('extracted_text', '')
                if not text:
                    continue
                is_dup = any(
                    self._text_similarity(text, e.get('extracted_text', '')) >= self.similarity_threshold
                    for e in unique
                )
                if not is_dup:
                    unique.append(stmt)
        return {'statements': unique}

    # =========================================================================
    # AMENDMENTS
    # =========================================================================
    
    def merge_amendments(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deduplicate by claim number, keep most complete entry"""
        by_claim = {}
        for result in results:
            for amend in result.get('amended_claims', []):
                num = amend.get('claim_number')
                if num is None:
                    continue
                if num not in by_claim or self._amendment_score(amend) > self._amendment_score(by_claim[num]):
                    by_claim[num] = amend
        
        renumbering = []
        renum_keys = set()
        for result in results:
            for r in result.get('renumbering', []):
                key = (r.get('old_number'), r.get('new_number'))
                if key not in renum_keys:
                    renum_keys.add(key)
                    renumbering.append(r)
        
        amendments = sorted(by_claim.values(), key=lambda a: a.get('claim_number', 0))
        return {'amended_claims': amendments, 'renumbering': renumbering}
    
    def _amendment_score(self, amend: Dict[str, Any]) -> int:
        score = 0
        if amend.get('current_text'):
            score += len(amend['current_text'])
        if amend.get('previous_text'):
            score += 1000
        if amend.get('change_summary'):
            score += 500
        if amend.get('added_limitations'):
            score += len(amend['added_limitations']) * 200
        return score

    # =========================================================================
    # RESTRICTION
    # =========================================================================
    
    def merge_restriction(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}
        
        all_groups = []
        group_names = set()
        elected_group = None
        elected_claims = []
        non_elected_claims = []
        traversed = False
        traversal_arguments = None
        linking_claims = []
        
        for result in results:
            for group in result.get('groups', []):
                name = group.get('group_name')
                if name and name not in group_names:
                    group_names.add(name)
                    all_groups.append(group)
            if not elected_group and result.get('elected_group'):
                elected_group = result['elected_group']
            if not elected_claims and result.get('elected_claims'):
                elected_claims = result['elected_claims']
            if not non_elected_claims and result.get('non_elected_claims'):
                non_elected_claims = result['non_elected_claims']
            if result.get('traversed'):
                traversed = True
            if not traversal_arguments and result.get('traversal_arguments'):
                traversal_arguments = result['traversal_arguments']
            if result.get('linking_claims'):
                linking_claims.extend(result['linking_claims'])
        
        return {
            'groups': all_groups,
            'elected_group': elected_group,
            'elected_claims': elected_claims,
            'non_elected_claims': non_elected_claims,
            'traversed': traversed,
            'traversal_arguments': traversal_arguments,
            'linking_claims': list(set(linking_claims)),
        }

    # =========================================================================
    # ALLOWANCE
    # =========================================================================
    
    def merge_allowance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}
        
        all_allowed = set()
        renumbering = []
        renum_keys = set()
        rfa = None
        
        for result in results:
            all_allowed.update(result.get('allowed_claims', []))
            for r in result.get('renumbering', []):
                key = (r.get('application_claim_number'), r.get('issued_claim_number'))
                if key not in renum_keys:
                    renum_keys.add(key)
                    renumbering.append(r)
            if not rfa and result.get('reasons_for_allowance', {}).get('stated'):
                rfa = result['reasons_for_allowance']
        
        examiner_amendments = []
        for result in results:
            examiner_amendments.extend(result.get('examiner_amendments', []))
        
        conditions = set()
        for result in results:
            conditions.update(result.get('conditions', []))
        
        return {
            'allowed_claims': sorted(all_allowed),
            'renumbering': renumbering,
            'reasons_for_allowance': rfa or {'stated': False},
            'examiner_amendments': examiner_amendments,
            'conditions': list(conditions),
        }

    # =========================================================================
    # INTERVIEW
    # =========================================================================
    
    def merge_interview(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}
        
        interview_date = next((r['interview_date'] for r in results if r.get('interview_date')), None)
        
        all_participants = set()
        all_topics = set()
        agreements = []
        agreement_topics = set()
        suggestions = []
        seen_suggestions = set()
        proposed = []
        
        for result in results:
            all_participants.update(result.get('participants', []))
            all_topics.update(result.get('topics_discussed', []))
            for agr in result.get('agreements_reached', []):
                topic = agr.get('topic', '')
                if topic and topic not in agreement_topics:
                    agreement_topics.add(topic)
                    agreements.append(agr)
            for sug in result.get('examiner_suggestions', []):
                if sug and sug not in seen_suggestions:
                    seen_suggestions.add(sug)
                    suggestions.append(sug)
            proposed.extend(result.get('proposed_amendments', []))
        
        return {
            'interview_date': interview_date,
            'participants': list(all_participants),
            'topics_discussed': list(all_topics),
            'agreements_reached': agreements,
            'examiner_suggestions': suggestions,
            'proposed_amendments': proposed,
        }

    # =========================================================================
    # TERMINAL DISCLAIMER
    # =========================================================================
    
    def merge_terminal_disclaimer(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        has_td = any(r.get('has_terminal_disclaimer', False) for r in results)
        
        disclaimed = []
        patent_numbers = set()
        for result in results:
            for patent in result.get('disclaimed_patents', []):
                num = patent.get('patent_number')
                if num and num not in patent_numbers:
                    patent_numbers.add(num)
                    disclaimed.append(patent)
        
        date = next((r['disclaimer_date'] for r in results if r.get('disclaimer_date')), None)
        reason = next((r['reason'] for r in results if r.get('reason')), None)
        
        return {
            'has_terminal_disclaimer': has_td,
            'disclaimed_patents': disclaimed,
            'disclaimer_date': date,
            'reason': reason,
        }

    # =========================================================================
    # MEANS-PLUS-FUNCTION
    # =========================================================================
    
    def merge_means_plus_function(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        mpf_elements = []
        mpf_keys = set()
        for result in results:
            for mpf in result.get('means_plus_function_elements', []):
                key = (mpf.get('claim_number'), mpf.get('element_text', '')[:50])
                if key not in mpf_keys:
                    mpf_keys.add(key)
                    mpf_elements.append(mpf)
        
        stmts = []
        stmt_texts = set()
        for result in results:
            for stmt in result.get('examiner_statements_on_112f', []):
                text = stmt.get('statement', '')[:50]
                if text and text not in stmt_texts:
                    stmt_texts.add(text)
                    stmts.append(stmt)
        
        return {
            'means_plus_function_elements': mpf_elements,
            'examiner_statements_on_112f': stmts,
        }

    # =========================================================================
    # COMPREHENSIVE / GENERIC
    # =========================================================================
    
    def merge_comprehensive(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}
        
        doc_type = next(
            (r['document_type'] for r in results if r.get('document_type') and r['document_type'] != "Unknown"),
            "Unknown",
        )
        doc_date = next((r['document_date'] for r in results if r.get('document_date')), None)
        
        claims_present = any(r.get('claims', {}).get('present', False) for r in results)
        all_claims = []
        if claims_present:
            claims_results = [{'claims': r.get('claims', {}).get('claim_list', [])} for r in results]
            all_claims = self.merge_claims(claims_results).get('claims', [])
        
        stmt_results = [{'statements': r.get('key_statements', [])} for r in results]
        merged_stmts = self.merge_statements(stmt_results)
        
        merged = {
            'document_type': doc_type,
            'document_date': doc_date,
            'claims': {'present': claims_present, 'claim_list': all_claims},
            'key_statements': merged_stmts.get('statements', []),
            'summary': results[0].get('summary', '') if results else '',
        }
        
        for key in ['rejections', 'arguments', 'restriction', 'allowance',
                     'terminal_disclaimer', 'means_plus_function']:
            for result in results:
                if result.get(key, {}).get('present', False):
                    merged[key] = result[key]
                    break
        
        return merged

    # =========================================================================
    # HELPER
    # =========================================================================
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        t1 = ' '.join(text1.lower().split())
        t2 = ' '.join(text2.lower().split())
        return SequenceMatcher(None, t1, t2).ratio()


def create_chunker(config: Dict[str, Any]) -> DocumentChunker:
    """Create a DocumentChunker from config"""
    proc = config.get('processing', {})
    return DocumentChunker(
        max_chunk_chars=proc.get('max_chunk_chars', 28000),
        overlap_chars=proc.get('chunk_overlap', 1500),
    )
