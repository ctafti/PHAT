
"""
PHAT Report Generator
Generates comprehensive File History Review Reports
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import defaultdict

from .database import (
    DatabaseManager, Patent, Document, Claim, ClaimVersion,
    ProsecutionStatement, RejectionHistory, PriorityBreak,
    TerminalDisclaimer, RestrictionRequirement, FinalClaim,
    ValidityRisk, TermSynthesis, ClaimNarrative, PatentTheme,
    TermBoundary, PriorArtReference, ClaimVulnerabilityCard, ProsecutionMilestone,
)

logger = logging.getLogger(__name__)


def _int_sort_key(x):
    """Sort key that handles mixed str/int claim numbers."""
    try:
        return int(x)
    except (ValueError, TypeError):
        return float('inf')


def _safe_int(val):
    """Coerce a value to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# =============================================================================
# Workstream 1: Global Deduplication Registry (v2.2)
# =============================================================================
class ArgumentRegistry:
    """
    Tracks arguments that have been fully rendered in the report.
    Subsequent sections can reference them by short label instead of
    restating the full text.
    """

    def __init__(self, similarity_threshold: float = 0.5):
        self._rendered: list = []
        # Each entry: {
        #   'text_key': str (first 200 chars, lowered),
        #   'section': str,
        #   'prior_art': list[str],
        # }
        self._threshold = similarity_threshold

    def is_rendered(self, text: str, prior_art: list = None) -> Optional[str]:
        """
        Check if an argument substantially similar to `text` has already
        been rendered. Returns the section label for cross-reference if
        so, or None if this is new.
        """
        from difflib import SequenceMatcher
        text_key = (text or '')[:200].lower().strip()
        if not text_key:
            return None

        for entry in self._rendered:
            sim = SequenceMatcher(None, text_key, entry['text_key']).ratio()
            if sim >= self._threshold:
                return entry['section']
            # Also match if same prior-art + high partial overlap
            if prior_art and entry.get('prior_art'):
                shared_art = set(a.lower() for a in prior_art) & set(
                    a.lower() for a in entry['prior_art']
                )
                if shared_art and sim >= 0.4:
                    return entry['section']

        return None

    def register(self, text: str, section: str,
                 prior_art: list = None):
        """Mark an argument as fully rendered in the given section."""
        text_key = (text or '')[:200].lower().strip()
        if not text_key:
            return
        self._rendered.append({
            'text_key': text_key,
            'section': section,
            'prior_art': prior_art or [],
        })


# =============================================================================
# Workstream 2: Entity Normalization (v2.2)
# =============================================================================
def normalize_prior_art_references(
    prior_art_map: dict,
    all_reference_keys: list = None,
) -> dict:
    """
    Normalize prior art reference names to merge OCR variants and
    citation-style variants into canonical keys.

    Phase 1 — Surname extraction & fuzzy match:
      "Davan et al. (US 7,616,217)" → surname "Davan"
      "Dayan"                       → surname "Dayan"
      get_close_matches("Davan", ["Dayan"]) → merge under "Dayan"

    Phase 2 — Collapse citation-style variants:
      "Dayan", "Dayan et al.", "Dayan, Zvuloni" → merge under shortest
      that is a standalone surname.
    """
    from difflib import get_close_matches

    if not prior_art_map:
        return prior_art_map

    def _extract_surname(ref: str) -> str:
        """Extract the primary inventor surname from a citation."""
        ref = ref.strip()
        # Remove patent number parentheticals
        ref = re.sub(r'\(US[\s\d,/A-Z]+\)', '', ref).strip()
        # Remove "et al."
        ref = re.sub(r'\bet\s+al\.?\b', '', ref, flags=re.IGNORECASE).strip()
        # Take the first comma-separated or space-separated name
        parts = re.split(r'[,;]', ref)
        surname = parts[0].strip()
        # Remove trailing punctuation
        surname = surname.rstrip(' .,;')
        return surname

    # Build surname → [original_keys] map
    surname_map: Dict[str, List[str]] = {}
    for key in prior_art_map.keys():
        surname = _extract_surname(key)
        if not surname:
            continue
        surname_lower = surname.lower()
        surname_map.setdefault(surname_lower, []).append(key)

    # Phase 1: Fuzzy-match surnames
    canonical_surnames: Dict[str, str] = {}  # surname_lower → canonical
    processed = set()

    for surname_lower in sorted(surname_map.keys(), key=len):
        if surname_lower in processed:
            continue
        # Find close matches among all other surnames
        all_others = [s for s in surname_map.keys() if s != surname_lower
                      and s not in processed]
        matches = get_close_matches(
            surname_lower, all_others, n=5, cutoff=0.75
        )
        # The canonical form is the most-used, then longest
        group = [surname_lower] + [m for m in matches]
        canonical = max(
            group,
            key=lambda s: (len(surname_map.get(s, [])), len(s))
        )
        for s in group:
            canonical_surnames[s] = canonical
            processed.add(s)

    # Phase 2: Merge original keys under canonical surnames
    key_to_canonical: Dict[str, str] = {}

    for surname_lower, original_keys in surname_map.items():
        canon_surname = canonical_surnames.get(surname_lower, surname_lower)
        # Gather all original keys for this canonical surname
        all_keys_for_canon = []
        for sl, oks in surname_map.items():
            if canonical_surnames.get(sl) == canon_surname:
                all_keys_for_canon.extend(oks)

        # Choose shortest key as display name
        display_key = min(all_keys_for_canon, key=len) if all_keys_for_canon else original_keys[0]

        for orig_key in original_keys:
            key_to_canonical[orig_key] = display_key

    # Rebuild the map under canonical keys
    merged: Dict[str, list] = {}
    for orig_key, entries in prior_art_map.items():
        canon = key_to_canonical.get(orig_key, orig_key)
        merged.setdefault(canon, []).extend(entries)

    return merged


class ReportGenerator:
    """Generates File History Review Reports"""
    
    def __init__(self, db_manager: DatabaseManager, output_dir: str):
        self.db_manager = db_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize string for use in filenames"""
        if not name:
            return "unknown"
        name = re.sub(r'[\\/*?:"<>|]', '_', name)
        return name.strip(' .,')
        
    def generate_report(self, patent_id: str, format: str = "both") -> Dict[str, str]:
        """Generate the full report for a patent"""
        session = self.db_manager.get_session()
        
        try:
            patent = session.query(Patent).get(patent_id)
            if not patent:
                raise ValueError(f"Patent not found: {patent_id}")
            
            report_data = self._gather_report_data(session, patent)
            
            output_files = {}
            
            base_name = patent.patent_number or patent.application_number or 'analysis'
            safe_name = self._sanitize_filename(base_name)
            
            if format in ["markdown", "both"]:
                md_content = self._generate_markdown(report_data)
                md_path = self.output_dir / f"report_{safe_name}.md"
                md_path.write_text(md_content)
                output_files['markdown'] = str(md_path)
                
            if format in ["html", "both"]:
                html_content = self._generate_html(report_data)
                html_path = self.output_dir / f"report_{safe_name}.html"
                html_path.write_text(html_content)
                output_files['html'] = str(html_path)
            
            json_path = self.output_dir / f"data_{safe_name}.json"
            json_path.write_text(json.dumps(report_data, indent=2, default=str))
            output_files['json'] = str(json_path)
            
            return output_files
            
        finally:
            session.close()
    
    def _gather_report_data(self, session, patent: Patent) -> Dict[str, Any]:
        """Gather all data needed for the report"""
        
        # Get all claims
        claims = session.query(Claim).filter(Claim.patent_id == patent.id).all()
        
        # Get all documents
        documents = session.query(Document).filter(Document.patent_id == patent.id).order_by(Document.document_date).all()
        
        # Get all statements
        statements = session.query(ProsecutionStatement).join(Document).filter(
            Document.patent_id == patent.id
        ).all()
        
        # Get all rejections
        rejections = session.query(RejectionHistory).join(Document).filter(
            Document.patent_id == patent.id
        ).all()
        
        # Get terminal disclaimers
        terminal_disclaimers = session.query(TerminalDisclaimer).filter(
            TerminalDisclaimer.patent_id == patent.id
        ).all()
        
        # Get restriction requirements
        restrictions = session.query(RestrictionRequirement).filter(
            RestrictionRequirement.patent_id == patent.id
        ).all()
        
        # Get priority breaks
        priority_breaks = session.query(PriorityBreak).filter(
            PriorityBreak.patent_id == patent.id
        ).all()
        
        # Get final claims
        final_claims = session.query(FinalClaim).filter(
            FinalClaim.patent_id == patent.id
        ).order_by(FinalClaim.claim_number).all()
        
        # Get Layer 2/3 synthesis data
        validity_risks = session.query(ValidityRisk).filter(
            ValidityRisk.patent_id == patent.id
        ).all()
        
        term_syntheses = session.query(TermSynthesis).filter(
            TermSynthesis.patent_id == patent.id
        ).all()
        
        claim_narratives = session.query(ClaimNarrative).filter(
            ClaimNarrative.patent_id == patent.id
        ).all()
        
        # Get patent themes (Improvement A)
        patent_themes = session.query(PatentTheme).filter(
            PatentTheme.patent_id == patent.id
        ).all()
        
        # Get term boundaries
        term_boundaries = session.query(TermBoundary).filter(
            TermBoundary.patent_id == patent.id
        ).all()
        
        # Get consolidated prior art references
        prior_art_refs = session.query(PriorArtReference).filter(
            PriorArtReference.patent_id == patent.id
        ).all()
        
        # Get vulnerability cards
        vulnerability_cards = session.query(ClaimVulnerabilityCard).filter(
            ClaimVulnerabilityCard.patent_id == patent.id
        ).all()
        
        #  Get prosecution milestones
        milestones = session.query(ProsecutionMilestone).filter(
            ProsecutionMilestone.patent_id == patent.id
        ).order_by(ProsecutionMilestone.date).all()
        
        # Build claim genealogy map
        genealogy = self._build_genealogy(claims, session)
        
        # Build estoppel events
        estoppel_events = self._identify_estoppel_events(statements, rejections)
        
        # Build claim element lexicon
        lexicon = self._build_lexicon(statements)
        
        # Build term narratives (synthesis keyed by term for report rendering)
        term_narratives = {}
        for ts in term_syntheses:
            term_narratives[ts.term] = {
                'summary': ts.narrative_summary,
                'status': ts.consistency_status,
                'contradiction_details': ts.contradiction_details,
                'source_statement_ids': ts.source_statement_ids or [],
            }
        
        # Build claim narrative map (keyed by claim number)
        claim_narrative_map = {}
        for cn in claim_narratives:
            claim_narrative_map[cn.claim_number] = {
                'evolution_summary': cn.evolution_summary,
                'turning_point_event': cn.turning_point_event,
            }
        
        # =====================================================================
        # Build a unified Prior Art Distinctions map from ALL sources
        # (statements + added_limitations on ClaimVersions)
        # =====================================================================
        prior_art_map = defaultdict(list)
        
        # Source 1: ProsecutionStatements with prior art citations
        for stmt in statements:
            cat = stmt.relevance_category or ''
            has_art = stmt.cited_prior_art and len(stmt.cited_prior_art) > 0
            is_distinction = (
                cat == 'Prior Art Distinction'
                or 'distinction' in cat.lower()
                or 'traversal' in cat.lower()
                or has_art
            )
            if is_distinction and has_art:
                for art_ref in stmt.cited_prior_art:
                    if art_ref and art_ref.lower() not in ('none', 'null', 'n/a', ''):
                        prior_art_map[art_ref].append({
                            'text': stmt.extracted_text,
                            'speaker': stmt.speaker,
                            'claims': stmt.affected_claims,
                            'element': stmt.claim_element_defined,
                            'category': cat,
                        })
        
        # Source 2: added_limitations on ClaimVersions (often has the richest data)
        for claim_data in [self._serialize_claim(c, session) for c in claims]:
            for v in claim_data.get('versions', []):
                for lim in (v.get('added_limitations') or []):
                    art = lim.get('likely_response_to_art', '')
                    if art and art.lower() not in ('none', 'null', 'n/a', '', 'not specified'):
                        prior_art_map[art].append({
                            'text': lim.get('text', ''),
                            'speaker': 'Applicant',
                            'claims': [claim_data['application_number']],
                            'element': lim.get('text', '')[:80],
                            'category': 'Amendment',
                            'argument': lim.get('applicant_argument_summary', ''),
                        })
        
        # =====================================================================
        # Collect Reasons for Allowance statements
        # =====================================================================
        reasons_for_allowance = [
            self._serialize_statement(s) for s in statements
            if s.relevance_category == 'Reasons for Allowance'
        ]
        
        # =====================================================================
        # Normalize prior art references to merge OCR
        # variants and citation-style variants into canonical keys.
        # =====================================================================
        prior_art_map = normalize_prior_art_references(dict(prior_art_map))
        
        # =====================================================================
        # Build a document page lookup for citation support
        # =====================================================================
        doc_page_map = {}
        doc_statements_map = defaultdict(list)  # Improvement C: doc_id -> list of serialized statements
        for doc in documents:
            if doc.page_start is not None:
                doc_page_map[doc.id] = {
                    'page_start': doc.page_start + 1,  # 0-indexed to 1-indexed
                    'page_end': (doc.page_end + 1) if doc.page_end is not None else None,
                    'type': doc.document_type,
                    'date': doc.document_date.isoformat() if doc.document_date else None,
                }
        
        # Improvement C: Build a map from document date to relevant arguments
        # This allows the claim evolution section to show "Why" for each amendment
        for stmt in statements:
            cat = stmt.relevance_category or ''
            if cat in ('Prior Art Distinction', 'Estoppel', 'Traversal',
                       'Definition/Interpretation', 'General Argument'):
                doc_date = None
                if stmt.document and stmt.document.document_date:
                    doc_date = stmt.document.document_date.isoformat()[:10]
                if doc_date:
                    doc_statements_map[doc_date].append({
                        'text': stmt.extracted_text,
                        'category': cat,
                        'prior_art': stmt.cited_prior_art,
                        'element': stmt.claim_element_defined,
                        'claims': stmt.affected_claims,
                    })
        
        # =====================================================================
        # Collect directionality constraints, categorical exclusions,
        # purpose characterizations, and core invention characterizations
        # =====================================================================
        directionality_constraints = []
        categorical_exclusions = []
        purpose_characterizations = []
        core_characterizations = []
        strategic_tensions = []
        
        for stmt in statements:
            ctx = stmt.context_summary or ''
            text = stmt.extracted_text or ''
            cat = stmt.relevance_category or ''
            
            # Strategic Tensions 
            if cat == 'Strategic Tension':
                strategic_tensions.append(self._serialize_statement(stmt))
            
            # Directionality Constraints 
            if '[DIRECTIONALITY CONSTRAINT]' in ctx:
                directionality_constraints.append({
                    'text': text,
                    'claims': stmt.affected_claims,
                    'prior_art': stmt.cited_prior_art,
                    'element': stmt.claim_element_defined,
                })
            
            # Categorical Exclusions 
            is_exclusion = (
                any(p in text.lower() for p in ['cannot be used', 'does not include', 'excludes',
                                                  'is not directed to', 'never includes', 'must not'])
                or (stmt.context_summary and '[NEGATIVE LIMITATION' in stmt.context_summary)
            )
            if is_exclusion and len(text) > 20:
                categorical_exclusions.append({
                    'text': text,
                    'claims': stmt.affected_claims,
                    'prior_art': stmt.cited_prior_art,
                    'element': stmt.claim_element_defined,
                })
            
            # Purpose/Function Characterizations (
            if '[PURPOSE/FUNCTION CHARACTERIZATION]' in ctx:
                purpose_characterizations.append({
                    'text': text,
                    'claims': stmt.affected_claims,
                    'element': stmt.claim_element_defined,
                })
            
            # Core Invention Characterization  — high-level descriptions of what
            # the invention IS that apply broadly (thematic distinctions, purpose characterizations)
            is_core = (
                '[PURPOSE/FUNCTION CHARACTERIZATION]' in ctx
                or 'serves as a' in text.lower()
                or 'in essence' in text.lower()
                or ('images' in text.lower() and ('hover' in text.lower() or 'integrated' in text.lower()))
                or ('catalog' in text.lower() and ('3d model' in text.lower() or '3D' in text))
            )
            if is_core and len(text) > 30:
                core_characterizations.append({
                    'text': text,
                    'claims': stmt.affected_claims,
                    'element': stmt.claim_element_defined,
                })
        
        # =====================================================================
        # Cross-claim limitation comparison (Scope Differentials)
        # Compare independent claims to find meaningful differences
        # =====================================================================
        scope_differentials = []
        if final_claims:
            independent_finals = [fc for fc in final_claims if fc.is_independent]
            if len(independent_finals) >= 2:
                scope_differentials = self._find_scope_differentials(independent_finals)
        
        return {
            'patent': {
                'patent_number': patent.patent_number,
                'application_number': patent.application_number,
                'title': patent.title,
                'filing_date': patent.filing_date,
                'issue_date': patent.issue_date,
                'priority_date': patent.priority_date,
                'has_final_claims': patent.has_final_claims
            },
            'claims': [self._serialize_claim(c, session) for c in claims],
            'final_claims': [self._serialize_final_claim(fc) for fc in final_claims],
            'documents': [self._serialize_document(d) for d in documents],
            'statements': [self._serialize_statement(s) for s in statements],
            'rejections': [self._serialize_rejection(r) for r in rejections],
            'terminal_disclaimers': [self._serialize_td(td) for td in terminal_disclaimers],
            'restrictions': [self._serialize_restriction(r) for r in restrictions],
            'priority_breaks': [self._serialize_priority_break(pb) for pb in priority_breaks],
            'genealogy': genealogy,
            'estoppel_events': estoppel_events,
            'lexicon': lexicon,
            # Layer 2/3 Synthesis data
            'validity_risks': [self._serialize_validity_risk(vr) for vr in validity_risks],
            'term_narratives': term_narratives,
            'claim_narratives': claim_narrative_map,
            'prior_art_map': dict(prior_art_map),
            'reasons_for_allowance': reasons_for_allowance,
            'doc_page_map': doc_page_map,
            'doc_statements_map': dict(doc_statements_map),
            'directionality_constraints': directionality_constraints,
            'categorical_exclusions': categorical_exclusions,
            'purpose_characterizations': purpose_characterizations,
            'core_characterizations': core_characterizations,
            'strategic_tensions': strategic_tensions,
            'scope_differentials': scope_differentials,
            'patent_themes': [self._serialize_patent_theme(pt) for pt in patent_themes],
            'term_boundaries': self._group_term_boundaries(term_boundaries),
            'prior_art_refs': [self._serialize_prior_art_ref(par) for par in prior_art_refs],
            'vulnerability_cards': {vc.claim_number: self._serialize_vulnerability_card(vc) for vc in vulnerability_cards},
            'milestones': [{'type': m.milestone_type, 'date': m.date.isoformat() if m.date else None, 'context': m.context} for m in milestones],
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _serialize_claim(self, claim: Claim, session) -> Dict:
        """Serialize a claim for the report"""
        versions = session.query(ClaimVersion).filter(
            ClaimVersion.claim_id == claim.id
        ).order_by(ClaimVersion.version_number).all()
        
        return {
            'application_number': claim.application_claim_number,
            'issued_number': claim.final_issued_number,
            'mapped_final_claim': claim.mapped_final_claim_number,
            'mapping_confidence': claim.mapping_confidence,
            'is_independent': claim.is_independent,
            'status': claim.current_status,
            'is_means_plus_function': claim.is_means_plus_function,
            'mpf_elements': claim.means_plus_function_elements,
            'is_elected': claim.is_elected,
            'versions': [{
                'version': v.version_number,
                'text': v.claim_text,
                'change_summary': v.change_summary,
                'date': v.date_of_change.isoformat() if v.date_of_change else None,
                'is_substantive': v.is_normalized_change,
                'added_limitations': v.added_limitations
            } for v in versions]
        }
    
    def _serialize_final_claim(self, fc: FinalClaim) -> Dict:
        """Serialize a final claim for the report"""
        return {
            'number': fc.claim_number,
            'text': fc.claim_text,
            'is_independent': fc.is_independent,
            'depends_on': fc.depends_on,
            'claim_type': fc.claim_type,  #Method/System/CRM/Apparatus/etc.
            'mapped_app_claim_id': fc.mapped_app_claim_id,
            'mapping_confidence': fc.mapping_confidence
        }
    
    def _serialize_document(self, doc: Document) -> Dict:
        return {
            'type': doc.document_type,
            'date': doc.document_date.isoformat() if doc.document_date else None,
            'pages': f"{doc.page_start + 1}-{doc.page_end + 1}" if doc.page_start is not None else None,
            'is_high_priority': doc.is_high_priority
        }
    
    def _serialize_statement(self, stmt: ProsecutionStatement) -> Dict:
        return {
            'speaker': stmt.speaker,
            'text': stmt.extracted_text,
            'category': stmt.relevance_category,
            'element_defined': stmt.claim_element_defined,
            'affected_claims': stmt.affected_claims,
            'prior_art': stmt.cited_prior_art,
            'is_acquiesced': stmt.is_acquiesced,
            'traversal_present': stmt.traversal_present
        }
    
    def _serialize_rejection(self, rej: RejectionHistory) -> Dict:
        return {
            'claim_number': rej.claim_number,
            'type': rej.rejection_type,
            'statutory_basis': rej.statutory_basis,
            'prior_art': rej.cited_prior_art,
            'elements': rej.rejected_claim_elements,
            'rationale': rej.rejection_rationale,
            'is_final': rej.is_final,
            'is_overcome': rej.is_overcome
        }
    
    def _serialize_td(self, td: TerminalDisclaimer) -> Dict:
        return {
            'disclaimed_patent': td.disclaimed_patent,
            'date': td.disclaimer_date.isoformat() if td.disclaimer_date else None,
            'reason': td.reason
        }
    
    def _serialize_restriction(self, restr: RestrictionRequirement) -> Dict:
        return {
            'date': restr.restriction_date.isoformat() if restr.restriction_date else None,
            'groups': restr.groups,
            'elected_group': restr.elected_group,
            'elected_claims': restr.elected_claims,
            'non_elected_claims': restr.non_elected_claims,
            'traversed': restr.traversed
        }
    
    def _serialize_priority_break(self, pb: PriorityBreak) -> Dict:
        return {
            'element': pb.feature_element,
            'status': pb.support_status,
            'notes': pb.analysis_notes
        }
    
    def _serialize_validity_risk(self, vr: ValidityRisk) -> Dict:
        return {
            'risk_type': vr.risk_type,
            'severity': vr.severity,
            'description': vr.description,
            'reasoning': vr.reasoning,
            'affected_claims': vr.affected_claims,
        }
    
    def _serialize_patent_theme(self, pt: PatentTheme) -> Dict:
        return {
            'title': pt.title,
            'summary': pt.summary,
            'key_arguments': pt.key_arguments or [],
            'key_amendments': pt.key_amendments or [],
            'prior_art_distinguished': pt.prior_art_distinguished or [],
            'affected_claims': pt.affected_claims or [],
            'estoppel_significance': pt.estoppel_significance,
            'metaphors_or_analogies': pt.metaphors_or_analogies or [],
        }
    
    @staticmethod
    def _group_term_boundaries(boundaries) -> Dict[str, List[Dict]]:
        """Group term boundaries by term for report rendering."""
        grouped = {}
        for tb in boundaries:
            term = tb.term or "Unknown"
            if term not in grouped:
                grouped[term] = []
            grouped[term].append({
                'boundary_type': tb.boundary_type,
                'example_text': tb.example_text,
                'source_text': tb.source_text,
            })
        return grouped
    
    @staticmethod
    def _serialize_prior_art_ref(par) -> Dict:
        """Serialize a PriorArtReference for report data."""
        return {
            'canonical_name': par.canonical_name,
            'patent_or_pub_number': par.patent_or_pub_number,
            'applied_basis': par.applied_basis,
            'affected_claims': par.affected_claims or [],
            'key_teachings': par.key_teachings or [],
            'key_deficiencies': par.key_deficiencies or [],
            'is_overcome': par.is_overcome,
        }
    
    @staticmethod
    def _serialize_vulnerability_card(vc) -> Dict:
        """erialize a ClaimVulnerabilityCard for report data."""
        return {
            'claim_number': vc.claim_number,
            'must_practice': vc.must_practice or [],
            'categorical_exclusions': vc.categorical_exclusions or [],
            'estoppel_bars': vc.estoppel_bars or [],
            'indefiniteness_targets': vc.indefiniteness_targets or [],
            'design_around_paths': vc.design_around_paths or [],
            'overall_vulnerability': vc.overall_vulnerability,
        }
    
    @staticmethod
    def _is_procedural_estoppel_text(text: str) -> bool:
        """Issue 1 fix: Check if estoppel text is purely procedural.
        
        Procedural statements like 'Applicant would like to discuss the sufficiency
        of the attached Draft Amendment' should not be flagged as estoppel events.
        """
        if not text:
            return True
        text_lower = text.lower().strip()
        procedural_phrases = [
            'would like to discuss', 'sufficiency of the attached',
            'request for interview', 'please call', 'thank you for',
            'acknowledge receipt', 'hereby submitted', 'respectfully submitted',
            'submitting this response', 'entry of this', 'authorized to charge',
            'petition to', 'extension of time', 'herewith transmits',
        ]
        if any(phrase in text_lower for phrase in procedural_phrases):
            # Check it doesn't also contain substantive content
            substantive = ['prior art', 'rejection', 'limitation', 'does not teach',
                          'fails to disclose', 'distinguishes', 'patentable over']
            if not any(sig in text_lower for sig in substantive):
                return True
        if len(text_lower) < 30:
            return True
        return False

    def _build_genealogy(self, claims: List[Claim], session) -> Dict:
        """Build claim genealogy mapping"""
        genealogy = {
            'app_to_issued': {},
            'issued_to_app': {},
            'cancelled': [],
            'withdrawn': [],
            'claim_families': defaultdict(list)
        }
        
        for claim in claims:
            app_num = claim.application_claim_number
            issued_num = claim.final_issued_number
            
            if claim.current_status == 'Cancelled':
                genealogy['cancelled'].append(app_num)
            elif claim.current_status == 'Withdrawn - Non-Elected':
                genealogy['withdrawn'].append(app_num)
            elif issued_num:
                genealogy['app_to_issued'][app_num] = issued_num
                genealogy['issued_to_app'][issued_num] = app_num
            
            if claim.family_tree_id:
                genealogy['claim_families'][claim.family_tree_id].append(app_num)
        
        return genealogy
    
    def _identify_estoppel_events(self, statements: List[ProsecutionStatement], 
                                   rejections: List[RejectionHistory]) -> List[Dict]:
        """Identify potential prosecution history estoppel events.
        
        Fix #2: Amendments made in response to prior art rejections that add new
        claim language are HIGH risk by default (Festo presumptive estoppel),
        not medium. The previous heuristic downgraded everything with a traversal
        to 'medium', which is incorrect — traversals don't avoid Festo estoppel
        when accompanied by narrowing amendments.
        """
        events = []
        
        # Build a set of prior art references from rejections for Festo analysis
        rejection_art_refs = set()
        for rej in rejections:
            if rej.cited_prior_art:
                if isinstance(rej.cited_prior_art, list):
                    for art in rej.cited_prior_art:
                        ref = art.get('reference', art) if isinstance(art, dict) else str(art)
                        rejection_art_refs.add(ref.lower().strip())
                elif isinstance(rej.cited_prior_art, str):
                    rejection_art_refs.add(rej.cited_prior_art.lower().strip())
        
        estoppel_categories = {
            'Prior Art Distinction', 'Estoppel', 'scope_limitation',
            'Traversal', 'General Argument',
        }
        
        for stmt in statements:
            cat = stmt.relevance_category or ''
            if cat in estoppel_categories or 'distinction' in cat.lower():
                # Issue 1 fix: Skip purely procedural statements
                text = stmt.extracted_text or ''
                if self._is_procedural_estoppel_text(text):
                    continue
                
                # Determine risk level with Festo-aware heuristic
                # If the statement cites prior art that was used in a rejection,
                # this is a Festo presumptive estoppel situation — HIGH risk
                risk_level = 'medium'  # default
                recapture_note = None
                
                cited_art = stmt.cited_prior_art or []
                cites_rejection_art = any(
                    (art.lower().strip() in rejection_art_refs)
                    for art in cited_art
                    if art and art.lower() not in ('none', 'null', 'n/a', '')
                )
                
                if cites_rejection_art:
                    # Festo: amendment narrowing in response to prior art rejection
                    # creates presumptive estoppel — HIGH risk regardless of traversal
                    risk_level = 'high'
                    art_names = ', '.join(a for a in cited_art if a and a.lower() not in ('none', 'null', 'n/a', ''))
                    recapture_note = (
                        f"Festo presumptive estoppel: Narrowing amendment made to "
                        f"overcome rejection based on {art_names}. Prosecution history "
                        f"estoppel may bar recapture of surrendered scope via doctrine "
                        f"of equivalents."
                    )
                elif not stmt.traversal_present:
                    # No traversal and no specific art — still high risk (acquiescence)
                    risk_level = 'high'
                
                event = {
                    'type': 'argument',
                    'text': stmt.extracted_text,
                    'claims': stmt.affected_claims,
                    'element': stmt.claim_element_defined,
                    'prior_art': stmt.cited_prior_art,
                    'risk_level': risk_level,
                }
                if recapture_note:
                    event['recapture_analysis'] = recapture_note
                events.append(event)
        
        return events
    
    def _build_lexicon(self, statements: List[ProsecutionStatement]) -> Dict[str, List[Dict]]:
        """Build claim element lexicon from definitions, filtering out generic terms (Gap 7)
        and clustering near-duplicate term names (Gap 7b)."""
        from difflib import SequenceMatcher
        
        GENERIC_TERMS = {
            'not specified', 'claims generally', 'amended claims', 'invention',
            'the invention', 'general', 'claims', 'invention (general)',
            'claim', 'the claims', 'all claims', 'independent claims',
            'dependent claims', 'the claim', 'n/a', 'none', 'not applicable',
            'the amendment', 'amendment', 'amendments', 'the application',
        }
        
        raw_lexicon = defaultdict(list)
        
        for stmt in statements:
            element = stmt.claim_element_defined
            if not element:
                continue
            if element.strip().lower() in GENERIC_TERMS:
                continue
            if len(element.strip()) < 3:
                continue
            
            raw_lexicon[element].append({
                'definition': stmt.extracted_text,
                'speaker': stmt.speaker,
                'is_acquiesced': stmt.is_acquiesced,
                'context': stmt.context_summary
            })
        
        # Cluster near-duplicate term names
        # E.g., "without alteration" and "presenting it without alteration" -> merge
        terms = list(raw_lexicon.keys())
        canonical_map = {}  # maps each term -> its canonical (shortest or first) form
        
        def _normalize_for_match(t: str) -> str:
            return re.sub(r'\s+', ' ', t.strip().lower())
        
        # Sort by length (prefer shorter terms as canonical)
        terms_sorted = sorted(terms, key=lambda t: len(t))
        
        for i, term_a in enumerate(terms_sorted):
            if term_a in canonical_map:
                continue
            canonical_map[term_a] = term_a  # self-canonical
            norm_a = _normalize_for_match(term_a)
            
            for term_b in terms_sorted[i+1:]:
                if term_b in canonical_map:
                    continue
                norm_b = _normalize_for_match(term_b)
                
                # Check: is one a substring of the other, or are they very similar?
                is_substring = norm_a in norm_b or norm_b in norm_a
                # Guard: for substring matches, the shorter term must be ≥40% of the longer
                # to avoid merging e.g. "it" into "without alteration"
                if is_substring:
                    shorter_len = min(len(norm_a), len(norm_b))
                    longer_len = max(len(norm_a), len(norm_b))
                    if shorter_len < longer_len * 0.4:
                        is_substring = False
                
                sim = SequenceMatcher(None, norm_a, norm_b).ratio()
                
                if is_substring or sim >= 0.75:
                    canonical_map[term_b] = term_a
        
        # Merge into canonical lexicon
        lexicon = defaultdict(list)
        for term, entries in raw_lexicon.items():
            canon = canonical_map.get(term, term)
            lexicon[canon].extend(entries)
        
        # Deduplicate entries within each term (by text prefix)
        for term in lexicon:
            seen = set()
            deduped = []
            for entry in lexicon[term]:
                key = entry['definition'][:100]
                if key not in seen:
                    seen.add(key)
                    deduped.append(entry)
            lexicon[term] = deduped
        
        return dict(lexicon)
    
    def _find_scope_differentials(self, independent_claims: List[FinalClaim]) -> List[Dict]:
        """Gap 3 fix: Compare independent claims to find meaningful scope differences.
        
        For example, if Claim 1 says 'two-dimensional geographic map' but Claim 2
        says only 'two-dimensional map' (without 'geographic'), this is a meaningful
        scope differential that should be flagged.
        
        Gap 8 fix: Filter out terms that reflect statutory category differences
        rather than substantive scope differences.
        """
        differentials = []
        
        # Gap 8: Blocklist of terms that reflect statutory category, not scope
        STATUTORY_CATEGORY_TERMS = {
            'non-transitory', 'computer-readable', 'machine-readable',
            'program storage medium', 'system comprising', 'method comprising',
            'apparatus comprising', 'method for', 'system for',
            'computer-implemented', 'processor-implemented',
        }
        
        # Extract key narrowing terms from each claim
        NARROWING_TERMS = [
            'geographic', 'ground truth', 'without alteration', 'non-transitory',
            'simultaneously', 'zooming', 'corresponding features', 'visual indicator',
            'transient foreground', 'pre-existing', 'pointing direction',
        ]
        
        for i, claim_a in enumerate(independent_claims):
            for claim_b in independent_claims[i+1:]:
                text_a = (claim_a.claim_text or '').lower()
                text_b = (claim_b.claim_text or '').lower()
                
                for term in NARROWING_TERMS:
                    # Gap 8: Skip statutory category terms
                    if term in STATUTORY_CATEGORY_TERMS:
                        continue
                    
                    in_a = term in text_a
                    in_b = term in text_b
                    
                    if in_a != in_b:
                        has_it = claim_a.claim_number if in_a else claim_b.claim_number
                        lacks_it = claim_b.claim_number if in_a else claim_a.claim_number
                        differentials.append({
                            'term': term,
                            'claim_with': has_it,
                            'claim_without': lacks_it,
                            'significance': (
                                f"Claim {has_it} requires '{term}' but Claim {lacks_it} does not. "
                                f"Claim {lacks_it} may have broader scope for this aspect."
                            ),
                        })
        
        return differentials

    def _group_independent_claims(self, final_claims: List[Dict], scope_differentials: List[Dict],
                                   claims_data: List[Dict]) -> List[Dict]:
        """Group independent claims into 'Claim Families' for the Strategic Summary.

        Claims that share >75% of the same added limitations are grouped together
        (e.g., Method/Apparatus pairs like Claims 1 & 2).  Claims with substantially
        different scope are placed in separate families (e.g., Claim 3).

        Returns a list of family dicts:
          { 'claims': [1, 2], 'label': 'Claims 1 & 2', 'type_hint': 'System / Method',
            'shared_limitations': [...], 'unique_limitations': {claim_num: [...]}, ... }
        """
        independent_finals = [fc for fc in final_claims if fc.get('is_independent')]
        if not independent_finals:
            return []

        # --- Build a per-claim limitation fingerprint ---
        # Map final claim number -> set of normalized limitation text snippets
        claim_limitations: Dict[int, set] = {}
        claim_type_hints: Dict[int, str] = {}

        for fc in independent_finals:
            fc_num = fc['number']
            fc_text = (fc.get('text') or '').lower()
            # Determine a rough type hint from claim preamble
            # Check CRM FIRST — Beauregard claims contain "method"/"process" in body (Bug Fix A)
            if 'medium' in fc_text[:120] or 'computer-readable' in fc_text[:120] or 'storage' in fc_text[:120] or 'non-transitory' in fc_text[:120]:
                claim_type_hints[fc_num] = 'Computer-Readable Medium'
            elif 'method' in fc_text[:80] or 'process' in fc_text[:80] or 'step of' in fc_text[:120]:
                claim_type_hints[fc_num] = 'Method'
            elif 'system' in fc_text[:80] or 'apparatus' in fc_text[:80] or 'device' in fc_text[:80]:
                claim_type_hints[fc_num] = 'System'
            else:
                claim_type_hints[fc_num] = 'Independent'

            # Gather added limitations from the claim versions that map to this final claim
            lim_set = set()
            for cd in claims_data:
                if cd.get('mapped_final_claim') == fc_num:
                    for v in cd.get('versions', []):
                        for lim in (v.get('added_limitations') or []):
                            lim_text = (lim.get('text') or '').strip().lower()
                            if lim_text and len(lim_text) > 5:
                                # Normalize: collapse whitespace, drop trailing punctuation
                                import re as _re
                                lim_norm = _re.sub(r'\s+', ' ', lim_text).strip(' .,;')
                                lim_set.add(lim_norm)
            claim_limitations[fc_num] = lim_set

        # --- Cluster claims by limitation overlap ---
        claim_nums = sorted(claim_limitations.keys(), key=_int_sort_key)
        assigned: Dict[int, int] = {}  # claim_num -> family_index
        families: List[Dict] = []

        for cn in claim_nums:
            if cn in assigned:
                continue
            # Start a new family
            family_members = [cn]
            assigned[cn] = len(families)

            for other_cn in claim_nums:
                if other_cn in assigned:
                    continue
                set_a = claim_limitations[cn]
                set_b = claim_limitations[other_cn]
                if not set_a and not set_b:
                    # Both empty — group if they share >75% of claim text tokens
                    text_a = next((fc.get('text', '') for fc in independent_finals if fc['number'] == cn), '')
                    text_b = next((fc.get('text', '') for fc in independent_finals if fc['number'] == other_cn), '')
                    tokens_a = set(text_a.lower().split()) - {'the', 'a', 'an', 'of', 'to', 'and', 'or', 'in', 'for', 'is', 'by', 'comprising', 'wherein'}
                    tokens_b = set(text_b.lower().split()) - {'the', 'a', 'an', 'of', 'to', 'and', 'or', 'in', 'for', 'is', 'by', 'comprising', 'wherein'}
                    if tokens_a and tokens_b:
                        overlap = len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
                    else:
                        overlap = 0.0
                elif not set_a or not set_b:
                    overlap = 0.0
                else:
                    overlap = len(set_a & set_b) / max(len(set_a), len(set_b))

                if overlap >= 0.75:
                    family_members.append(other_cn)
                    assigned[other_cn] = len(families)

            # Build family metadata
            member_nums = sorted(family_members, key=_int_sort_key)
            shared = set.intersection(*(claim_limitations[m] for m in member_nums)) if member_nums else set()
            unique: Dict[int, list] = {}
            for m in member_nums:
                diff = claim_limitations[m] - shared
                if diff:
                    unique[m] = sorted(diff)

            type_labels = [claim_type_hints.get(m, '') for m in member_nums]
            combined_type = ' / '.join(dict.fromkeys(t for t in type_labels if t))  # ordered dedup

            # Refinement A (v2.2): Supersedence Check — if Limitation A (e.g., "2D map")
            # is a substring of Limitation B (e.g., "2D geographic map"), filter out A
            # so the report shows only the final, most specific form.
            shared_list = sorted(shared)
            if len(shared_list) > 1:
                superseded = set()
                for i, lim_a in enumerate(shared_list):
                    for j, lim_b in enumerate(shared_list):
                        if i != j and lim_a in lim_b and lim_a != lim_b:
                            superseded.add(lim_a)
                if superseded:
                    shared_list = [l for l in shared_list if l not in superseded]
                    logger.debug(f"Supersedence check: filtered {len(superseded)} broad limitation(s)")

            if len(member_nums) == 1:
                label = f"Claim {member_nums[0]}"
            else:
                label = "Claims " + " & ".join(str(m) for m in member_nums)

            families.append({
                'claims': member_nums,
                'label': label,
                'type_hint': combined_type,
                'shared_limitations': shared_list,
                'unique_limitations': unique,
            })

        return families

    # =================================================================
    # Workstream 6: Mirror-Claim Condensation (v2.2)
    # =================================================================
    def _detect_mirror_claims(
        self, families: List[Dict], claims_data: List[Dict]
    ) -> Dict[int, int]:
        """
        Detect claims within the same family that share >75% of amendment
        events and arguments. Returns a map of secondary_claim → primary_claim
        for condensation.
        """
        mirrors: Dict[int, int] = {}  # secondary → primary

        for family in families:
            if len(family['claims']) < 2:
                continue

            # Build per-claim amendment fingerprint
            claim_fingerprints: Dict[int, List[str]] = {}
            for cn in family['claims']:
                cd = next(
                    (c for c in claims_data if c.get('mapped_final_claim') == cn),
                    None
                )
                if not cd:
                    continue
                fingerprint = []
                for v in cd.get('versions', []):
                    date = (v.get('date') or '')[:10]
                    lim_texts = sorted(
                        (lim.get('text') or '').lower()[:60]
                        for lim in (v.get('added_limitations') or [])
                    )
                    fingerprint.append(f"{date}:{'|'.join(lim_texts)}")
                claim_fingerprints[cn] = fingerprint

            # Compare pairs
            nums = sorted(claim_fingerprints.keys(), key=_int_sort_key)
            for i, cn_a in enumerate(nums):
                for cn_b in nums[i + 1:]:
                    if cn_b in mirrors:
                        continue
                    fp_a = claim_fingerprints.get(cn_a, [])
                    fp_b = claim_fingerprints.get(cn_b, [])
                    if not fp_a or not fp_b:
                        continue

                    # Calculate overlap
                    shared = len(set(fp_a) & set(fp_b))
                    total = max(len(fp_a), len(fp_b))
                    if total > 0 and shared / total >= 0.75:
                        mirrors[cn_b] = cn_a  # cn_b is the secondary

        return mirrors

    # =================================================================
    # Workstream 3: Event-Based Claim History Renderer (v2.2)
    # =================================================================
    def _render_claim_evolution_event_based(
        self, claim: dict, data: dict, registry: ArgumentRegistry
    ) -> str:
        """
        Render a claim's prosecution history as a sequence of Amendment
        events rather than a sequence of version snapshots.
        """
        md = ""
        versions = claim.get('versions', [])
        app_num = claim['application_number']

        # Phase 1: Group versions by date
        date_groups: Dict[str, List[dict]] = {}
        for v in versions:
            date = (v.get('date') or 'Unknown')[:10]
            date_groups.setdefault(date, []).append(v)

        prev_text = None  # Track previous version text for non-substantive detection

        # Phase 2: Render each date-group as one event
        for date, group_versions in date_groups.items():
            # Determine if this is the original filing
            if all(v['version'] == 1 for v in group_versions):
                md += f"**Original Filing ({date})**\n\n"
                text = (group_versions[0].get('text') or '')
                word_count = len(text.split())
                md += (
                    f"Originally recited "
                    f"({word_count} words). "
                    f"See Appendix for full text.\n\n"
                )
                prev_text = text
                continue

            # Collect all added_limitations across versions in this group
            all_limitations = []
            all_change_summaries = []
            is_substantive = False

            for v in group_versions:
                if v.get('added_limitations'):
                    all_limitations.extend(v['added_limitations'])
                    is_substantive = True
                if v.get('change_summary'):
                    all_change_summaries.append(v['change_summary'])
                if v.get('is_substantive'):
                    is_substantive = True

            # Skip non-substantive date groups (housekeeping only)
            if not is_substantive and not all_limitations:
                # Check if the text actually changed significantly
                texts = [v.get('text', '') for v in group_versions]
                current_text = texts[-1] if texts else ''
                if prev_text and current_text:
                    # Simple word-level diff
                    prev_words = set(prev_text.lower().split())
                    curr_words = set(current_text.lower().split())
                    diff_words = len(prev_words.symmetric_difference(curr_words))
                    if diff_words < 5:
                        prev_text = current_text
                        continue
                elif len(set(texts)) <= 1:
                    prev_text = texts[0] if texts else prev_text
                    continue

            # Check for is_housekeeping flag
            has_only_housekeeping = all_limitations and all(
                lim.get('is_housekeeping', False) for lim in all_limitations
            )
            if has_only_housekeeping:
                # Brief note for housekeeping amendments
                md += f"**Housekeeping Amendment ({date})**\n\n"
                for summary in all_change_summaries:
                    if summary:
                        md += f"- {summary}\n"
                md += "\n"
                prev_text = group_versions[-1].get('text', prev_text)
                continue

            # Determine event type label from documents
            event_label = "Amendment"
            for doc in data.get('documents', []):
                doc_date = (doc.get('date') or '')[:10]
                if doc_date == date:
                    doc_type = (doc.get('type') or '').lower()
                    if 'rce' in doc_type or 'continued' in doc_type:
                        event_label = "RCE + Amendment"
                    elif 'supplemental' in doc_type:
                        event_label = "Supplemental Amendment"
                    elif 'examiner' in doc_type and 'amendment' in doc_type:
                        event_label = "Examiner's Amendment"
                    elif 'interview' in doc_type:
                        event_label = "Examiner Interview"
                    break

            md += f"**{event_label} ({date})**\n\n"

            # Render each limitation as an Amendment/Argument pair
            if all_limitations:
                for lim in all_limitations:
                    if lim.get('is_housekeeping', False):
                        continue  # Skip housekeeping items in mixed groups

                    lim_text = (lim.get('text') or '').strip()
                    argument = (lim.get('applicant_argument_summary') or '').strip()
                    art = (lim.get('likely_response_to_art') or '').strip()
                    estoppel = (lim.get('estoppel_risk') or '').lower()

                    if lim_text:
                        md += f"- **Amendment:** Added \"{lim_text}\"\n"

                    if art and art.lower() not in ('none', 'n/a', '', 'not specified'):
                        md += f"  - **Distinguishes:** *{art}*"
                        if argument:
                            md += f" — {argument[:300]}"
                        md += "\n"
                    elif argument:
                        md += f"  - **Argument:** {argument[:300]}\n"

                    if estoppel in ('high',):
                        md += (
                            f"  - ⚠️ Prosecution history estoppel may bar "
                            f"recapture of the scope given up by this "
                            f"limitation.\n"
                        )

                    # Practical implications (Workstream 4)
                    if lim.get('practical_implications'):
                        for impl in lim['practical_implications']:
                            if impl:
                                md += f"  - 💡 *Implication:* {impl}\n"

                    md += "\n"

                    # Register this argument
                    registry.register(
                        argument or lim_text,
                        f"Claim {app_num} History",
                        [art] if art else None
                    )
            else:
                # Fallback for versions without structured limitations
                for summary in all_change_summaries:
                    if summary:
                        md += f"- **Change:** {summary}\n\n"

            # Update prev_text for next iteration
            prev_text = group_versions[-1].get('text', prev_text)

        return md

    # =================================================================
    # Workstream 5: Lexicon Quality Filter (v2.2)
    # =================================================================
    def _should_include_lexicon_entry(
        self, element: str, definitions: list,
        term_narratives: dict
    ) -> bool:
        """
        Determine if a lexicon entry has enough substance to include in
        the main report. Weak entries are moved to the appendix.
        """
        # Must have at least one Applicant statement
        has_applicant = any(
            d.get('speaker', '').lower() in ('applicant', 'appellant')
            for d in definitions
        )
        if not has_applicant:
            return False

        # Check synthesis narrative — skip if it says "only examiner"
        narrative = term_narratives.get(element, {})
        summary = (narrative.get('summary') or '').lower()
        if 'only the examiner' in summary or 'no applicant position' in summary:
            return False

        # Must have at least one definition longer than 50 chars
        has_substance = any(
            len(d.get('definition', '')) > 50 for d in definitions
        )
        return has_substance

    def _generate_strategic_summary(self, data: Dict) -> str:
        """Generate the human-style Strategic Summary section.

        This replaces the old process-oriented Executive Summary with an
        outcome-oriented, claim-centric summary that groups analysis by
        Claim Families and integrates scope constraints and risks inline.
        """
        md = ""

        # ── 0. Invention Thesis (single strongest core characterization) ──
        md += "### Invention Thesis\n\n"

        thesis_text = None
        # Strategic Enhancement C (v2.2): First check for conceptual_analogy statements
        # or core characterizations containing metaphor/essence keywords — these usually
        # capture the "spirit" of the invention better than dry technical definitions.
        for char in (data.get('core_characterizations') or []):
            text = (char.get('text') or '').strip()
            lower = text.lower()
            stype = (char.get('statement_type') or '').lower()
            if stype == 'conceptual_analogy' or any(
                kw in lower for kw in ['metaphor', 'essence', 'akin to', 'serves as',
                                        'acts like', 'functions as', 'analogous',
                                        'hover', 'ether', 'catalog', 'spirit']
            ):
                thesis_text = text
                break
        # Fallback: prefer a core characterization with distinction language
        if not thesis_text:
            for char in (data.get('core_characterizations') or []):
                text = (char.get('text') or '').strip()
                lower = text.lower()
                if any(kw in lower for kw in ['not', 'rather than', 'instead of', 'as opposed to',
                                               'unlike', 'distinct from', 'hover', 'integrated']):
                    thesis_text = text
                    break
        # Fallback: check patent_themes for metaphors_or_analogies
        if not thesis_text and data.get('patent_themes'):
            for theme in data['patent_themes']:
                metaphors = theme.get('metaphors_or_analogies') or []
                if metaphors and any(m for m in metaphors):
                    # Use the theme summary that contains the metaphor
                    thesis_text = theme.get('summary', '')
                    break
        # Fallback: strongest theme summary
        if not thesis_text and data.get('patent_themes'):
            thesis_text = data['patent_themes'][0].get('summary', '')
        # Fallback: first core characterization
        if not thesis_text and data.get('core_characterizations'):
            thesis_text = data['core_characterizations'][0].get('text', '')

        if thesis_text:
            # Truncate at sentence boundary around 2000 chars to prevent massive walls of text
            if len(thesis_text) > 2000:
                boundary = thesis_text[:2000].rfind('. ')
                if boundary > 1000:
                    thesis_text = thesis_text[:boundary + 1]
                else:
                    thesis_text = thesis_text[:2000] + '...'
            md += f"> **{thesis_text}**\n\n"
        else:
            md += "_No core invention characterization extracted._\n\n"

        # ── 1. Group independent claims into families ──
        families = self._group_independent_claims(
            data.get('final_claims', []),
            data.get('scope_differentials', []),
            data.get('claims', []),
        )

        if not families:
            md += "_No independent claims available for strategic grouping._\n\n"
            return md

        # Pre-build lookup helpers
        app_to_issued: Dict[int, int] = {}
        for c in data.get('claims', []):
            if c.get('mapped_final_claim') is not None:
                app_to_issued[c['application_number']] = c['mapped_final_claim']

        # ── 2. Per-Family Summary Blocks ──
        for family in families:
            member_nums = family['claims']  # issued/final claim numbers
            md += f"### {family['label']}"
            if family.get('type_hint'):
                md += f"  ({family['type_hint']})"
            md += "\n\n"

            # 2a. Core Concept — derive from the strongest theme affecting these claims
            relevant_themes = []
            for theme in (data.get('patent_themes') or []):
                affected = theme.get('affected_claims') or []
                # affected may be app claim numbers; map to final
                theme_issued = set()
                for ac in affected:
                    ac_i = _safe_int(ac) or ac
                    theme_issued.add(app_to_issued.get(ac_i, ac_i))
                if theme_issued & set(member_nums):
                    relevant_themes.append(theme)

            if relevant_themes:
                best_theme = relevant_themes[0]
                # Increased limit from 300 to 1000
                md += f"**Core Concept:** {best_theme.get('summary', '')[:1000]}\n\n"
                if best_theme.get('metaphors_or_analogies'):
                    metaphors = [m for m in best_theme['metaphors_or_analogies'] if m]
                    if metaphors:
                        md += f"*Key Characterizations:* {'; '.join(metaphors[:3])}\n\n"

            # 2b. Key Requirements (from added limitations + claim text key phrases)
            requirements = []
            for cn in member_nums:
                fc = next((f for f in data['final_claims'] if f['number'] == cn), None)
                if not fc:
                    continue
                claim_text = fc.get('text', '')
                for phrase in ['comprising', 'wherein', 'configured to', 'adapted to', 'such that']:
                    if phrase in claim_text.lower():
                        idx = claim_text.lower().index(phrase)
                        snippet = claim_text[idx:idx + 150].replace('\n', ' ').strip()
                        if snippet:
                            requirements.append((cn, snippet))

            if family.get('shared_limitations'):
                md += "**Key Requirements** (added during prosecution):\n"
                for lim in family['shared_limitations'][:8]:
                    md += f"  - {lim}\n"
                md += "\n"

            # Show unique limitations per claim within the family
            if family.get('unique_limitations'):
                for cn, uniques in family['unique_limitations'].items():
                    if uniques:
                        md += f"**Additional requirements specific to Claim {cn}:**\n"
                        for u in uniques[:5]:
                            md += f"  - {u}\n"
                        md += "\n"

            # 2b'. Practical Implications (v2.2 Workstream 4)
            implications = []
            for cd in data.get('claims', []):
                if cd.get('mapped_final_claim') not in member_nums:
                    continue
                for v in cd.get('versions', []):
                    for lim in (v.get('added_limitations') or []):
                        for impl in (lim.get('practical_implications') or []):
                            if impl and impl not in implications:
                                implications.append(impl)

            if implications:
                md += "**Practical Constraints:**\n"
                for impl in implications[:6]:
                    md += f"  - {impl}\n"
                md += "\n"

            # 2c. Negative Scope (categorical exclusions affecting these claims)
            claim_exclusions = [
                e for e in data.get('categorical_exclusions', [])
                if e.get('claims') and any(cn in (e.get('claims') or []) for cn in member_nums)
            ]
            # Also check if app-claim numbers hit
            for e in data.get('categorical_exclusions', []):
                mapped_issued = {app_to_issued.get(_safe_int(ac) or ac, _safe_int(ac) or ac) for ac in (e.get('claims') or [])}
                if mapped_issued & set(member_nums) and e not in claim_exclusions:
                    claim_exclusions.append(e)

            if claim_exclusions:
                md += "**Excludes:**\n"
                seen_excl = set()
                for excl in claim_exclusions[:5]:
                    key = (excl.get('text') or '')[:80]
                    if key in seen_excl:
                        continue
                    seen_excl.add(key)
                    # Removed truncation
                    md += f"  - ❌ {excl.get('text', '')}\n"
                md += "\n"

            # 2d. Risks (Shadow Examiner flags affecting these claims)
            claim_risks = []
            for risk in (data.get('validity_risks') or []):
                affected = risk.get('affected_claims') or []
                risk_issued = set()
                for ac in affected:
                    ac_i = _safe_int(ac) or ac
                    risk_issued.add(app_to_issued.get(ac_i, ac_i))
                if risk_issued & set(member_nums):
                    claim_risks.append(risk)
            # Also check if the risk has no specific claims (applies globally)
            for risk in (data.get('validity_risks') or []):
                if not risk.get('affected_claims') and risk not in claim_risks:
                    claim_risks.append(risk)

            if claim_risks:
                md += "**Risks:**\n"
                for risk in claim_risks[:5]:
                    severity = risk.get('severity', 'Unknown')
                    severity_icon = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(severity, '⚪')
                    # Removed truncation
                    md += f"  - {severity_icon} **{risk.get('risk_type', 'Unknown')}** ({severity}): {risk.get('description', '')}\n"
                md += "\n"

            # 2e. High-risk Estoppel affecting these claims
            claim_estoppel = []
            for e in data.get('estoppel_events', []):
                if e.get('risk_level') != 'high':
                    continue
                evt_issued = set()
                for ac in (e.get('claims') or []):
                    ac_i = _safe_int(ac) or ac
                    evt_issued.add(app_to_issued.get(ac_i, ac_i))
                if evt_issued & set(member_nums):
                    claim_estoppel.append(e)

            if claim_estoppel:
                md += "**Key Estoppel Commitments:**\n"
                seen_est = set()
                for est in claim_estoppel[:5]:
                    key = (est.get('text') or '')[:100]
                    if key in seen_est:
                        continue
                    seen_est.add(key)
                    # Gap 11: Never truncate estoppel commitment text
                    text = (est.get('text') or '')
                    art_refs = [a for a in (est.get('prior_art') or []) if a and a.lower() not in ('none', 'null', 'n/a', '')]
                    art_label = f" (vs. {', '.join(art_refs)})" if art_refs else ""
                    md += f"  - ⚠️ {text}{art_label}\n"
                md += "\n"

            # 2f. Directionality constraints
            claim_dirs = [
                d for d in data.get('directionality_constraints', [])
                if d.get('claims') and any(cn in (d.get('claims') or []) for cn in member_nums)
            ]
            if claim_dirs:
                md += "**Directionality Constraints:**\n"
                for dc in claim_dirs[:3]:
                    md += f"  - ↔️ {dc.get('text', '')}\n"
                md += "\n"

        # ── 3. Scope Differentials between families ──
        if data.get('scope_differentials'):
            md += "### Scope Differentials Between Claim Families\n\n"
            md += "| Term | Present In | Absent From | Significance |\n"
            md += "|------|-----------|-------------|-------------|\n"
            for diff in data['scope_differentials']:
                # Increased limit for table cells
                md += (f"| \"{diff['term']}\" | Claim {diff['claim_with']} | "
                       f"Claim {diff['claim_without']} | {diff['significance'][:500]} |\n")
            md += "\n"

        # ── 4. Terminal Disclaimers (brief) ──
        if data.get('terminal_disclaimers'):
            md += "### Terminal Disclaimers\n"
            for td in data['terminal_disclaimers']:
                md += f"- Disclaimed over: {td.get('disclaimed_patent', 'N/A')} ({td.get('date', 'N/A')})\n"
            md += "\n"

        # ── 5. Priority Flags (brief) ──
        if data.get('priority_breaks'):
            md += "### Priority Support Flags\n"
            for pb in data['priority_breaks']:
                md += f"- **{pb.get('element', 'N/A')}**: {pb.get('status', 'N/A')}\n"
            md += "\n"

        return md

    def _generate_markdown(self, data: Dict) -> str:
        """Generate Markdown report"""
        patent = data['patent']
        
        md = f"""# File History Review Report

**Patent/Application:** {patent.get('patent_number') or patent.get('application_number') or 'N/A'}  
**Title:** {patent.get('title') or 'N/A'}  
**Filing Date:** {patent.get('filing_date') or 'N/A'}  
**Report Generated:** {data['generated_at']}

---

"""
        
        # =====================================================================
        # I. STRATEGIC SUMMARY (Human-style, claim-centric)
        # =====================================================================
        md += "## I. Strategic Summary\n\n"
        md += self._generate_strategic_summary(data)
        md += "\n---\n\n"

        # Final Claims Section (Ground Truth)
        if data['final_claims']:
            md += "## Final Issued Claims (Ground Truth)\n\n"
            md += "The following claims represent the final issued claims from the patent:\n\n"
            
            for fc in data['final_claims']:
                claim_type = "Independent" if fc['is_independent'] else f"Depends on claim {fc['depends_on']}"
                type_label = f" [{fc['claim_type']}]" if fc.get('claim_type') and fc['claim_type'] != 'Unknown' else ""
                md += f"### Claim {fc['number']} ({claim_type}{type_label})\n\n"
                md += f"{fc['text']}\n\n"
            
            md += "---\n\n"

        # =====================================================================
        # v2.2: Initialize the Global Deduplication Registry (Workstream 1)
        # The registry tracks which arguments have been fully rendered so
        # subsequent sections can cross-reference instead of restating.
        # =====================================================================
        registry = ArgumentRegistry()

        # =====================================================================
        # II. SPECIFIC LIMITATIONS & DEFINITIONS (v2.2 Workstream 5 — promoted Lexicon)
        # Placed before claim histories so readers understand key terms first.
        # =====================================================================
        md += "## II. Specific Limitations & Definitions\n\n"
        md += (
            "The following terms were defined or characterized during prosecution. "
            "These definitions constrain claim scope and are binding under "
            "prosecution history estoppel.\n\n"
        )

        if data['lexicon']:
            # Filter lexicon entries for the main report; weak entries go to Appendix B
            main_lexicon_entries = {}
            appendix_lexicon_entries = {}
            for element, definitions in data['lexicon'].items():
                if self._should_include_lexicon_entry(element, definitions, data.get('term_narratives', {})):
                    main_lexicon_entries[element] = definitions
                else:
                    appendix_lexicon_entries[element] = definitions

            if main_lexicon_entries:
                for element, definitions in main_lexicon_entries.items():
                    md += f"### {element}\n\n"

                    # Layer 1: Synthesis narrative (if available)
                    narrative = data.get('term_narratives', {}).get(element)
                    if narrative:
                        md += f"**Synthesis:** {narrative['summary']}\n\n"

                        # Consistency warning
                        status = narrative.get('status', 'Unknown')
                        if status == 'Contradictory':
                            md += f"> ⚠️ **Inconsistency Detected:** {narrative.get('contradiction_details', 'See raw data below.')}\n\n"
                        elif status == 'Evolving':
                            md += f"> ℹ️ **Definition Evolved:** {narrative.get('contradiction_details', 'The definition was refined over time.')}\n\n"

                    # Layer 2: Raw data drill-down
                    md += "**Raw Evidence:**\n\n"
                    for defn in definitions:
                        speaker = defn.get('speaker', 'Unknown')
                        acquiesced = " ✓ (Acquiesced)" if defn.get('is_acquiesced') else ""
                        md += f"**{speaker}**{acquiesced}:\n"
                        md += f"> {defn.get('definition', 'N/A')}\n\n"

                        # Register definitions in the ArgumentRegistry
                        registry.register(
                            defn.get('definition', ''),
                            f"Section II: {element}",
                        )

                if appendix_lexicon_entries:
                    md += f"*{len(appendix_lexicon_entries)} additional term(s) with weaker evidence are listed in Appendix B.*\n\n"
            else:
                md += "No claim term definitions with sufficient applicant evidence extracted. See Appendix B for all raw entries.\n\n"
        else:
            md += "No claim term definitions extracted.\n\n"

        # Gap 1: Term Boundaries (actionable scope markers)
        term_boundaries = data.get('term_boundaries', {})
        if term_boundaries:
            md += "### Term Scope Boundaries\n\n"
            md += (
                "The following boundaries define what specific claim terms "
                "include and exclude, based on prosecution statements:\n\n"
            )
            for term, bounds in term_boundaries.items():
                md += f"**{term}**\n\n"
                includes = [b for b in bounds if b['boundary_type'] == 'includes']
                excludes = [b for b in bounds if b['boundary_type'] == 'excludes']
                if includes:
                    md += "  *Includes:*\n"
                    for b in includes:
                        md += f"  - {b['example_text']}"
                        if b.get('source_text'):
                            md += f" — *\"{b['source_text'][:120]}\"*"
                        md += "\n"
                if excludes:
                    md += "  *Excludes:*\n"
                    for b in excludes:
                        md += f"  - {b['example_text']}"
                        if b.get('source_text'):
                            md += f" — *\"{b['source_text'][:120]}\"*"
                        md += "\n"
                md += "\n"

        md += "---\n\n"

        # Gap 5: Per-Claim Vulnerability Cards
        vuln_cards = data.get('vulnerability_cards', {})
        if vuln_cards:
            md += "## Claim Vulnerability Assessment\n\n"
            md += (
                "Per-claim vulnerability cards synthesizing estoppel bars, "
                "categorical exclusions, and design-around paths.\n\n"
            )
            for claim_num in sorted(vuln_cards.keys(), key=_int_sort_key):
                card = vuln_cards[claim_num]
                overall = card.get('overall_vulnerability', 'Unknown')
                emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(overall, '⚪')
                md += f"### Claim {claim_num} — {emoji} {overall} Vulnerability\n\n"
                if card.get('must_practice'):
                    md += "**Must-Practice Elements:**\n"
                    for item in card['must_practice']:
                        md += f"- {item}\n"
                    md += "\n"
                if card.get('categorical_exclusions'):
                    md += "**Categorical Exclusions (Estoppel):**\n"
                    for item in card['categorical_exclusions']:
                        md += f"- {item}\n"
                    md += "\n"
                if card.get('estoppel_bars'):
                    md += "**Estoppel Bars:**\n"
                    for item in card['estoppel_bars']:
                        md += f"- {item}\n"
                    md += "\n"
                if card.get('indefiniteness_targets'):
                    md += "**Indefiniteness Targets:**\n"
                    for item in card['indefiniteness_targets']:
                        md += f"- {item}\n"
                    md += "\n"
                if card.get('design_around_paths'):
                    md += "**Design-Around Paths:**\n"
                    for item in card['design_around_paths']:
                        md += f"- {item}\n"
                    md += "\n"
            md += "---\n\n"

        # =====================================================================
        # III. DETAILED ANALYSIS (drill-down sections)
        # =====================================================================
        md += "## III. Detailed Prosecution Record\n\n"

        # -----------------------------------------------------------------
        # A. Reasons for Allowance
        # -----------------------------------------------------------------
        md += "### A. Reasons for Allowance\n"
        if data.get('reasons_for_allowance'):
            for rfa in data['reasons_for_allowance']:
                # Removed truncation
                md += f"\n> {rfa.get('text', 'N/A')}\n"
                if rfa.get('affected_claims'):
                    md += f">\n> *Affected Claims:* {rfa['affected_claims']}\n"

                # Cross-reference: find which amendment added the distinguishing feature
                rfa_text_lower = (rfa.get('text') or '').lower()
                for claim_data in data['claims']:
                    for v in claim_data.get('versions', []):
                        if v.get('version', 1) == 1:
                            continue
                        for lim in (v.get('added_limitations') or []):
                            lim_text = (lim.get('text') or '').lower()
                            if lim_text and len(lim_text) > 10:
                                lim_words = set(lim_text.split()) - {'the', 'a', 'an', 'of', 'to', 'in', 'for', 'and', 'or', 'is', 'by'}
                                match_count = sum(1 for w in lim_words if w in rfa_text_lower)
                                if len(lim_words) > 0 and match_count / len(lim_words) > 0.5:
                                    md += (
                                        f">\n> 🔗 *This feature was added in Claim "
                                        f"{claim_data['application_number']} "
                                        f"({v.get('date', 'Unknown')[:10] if v.get('date') else 'Unknown'})*\n"
                                    )
                                    break
        else:
            md += "\nNo Reasons for Allowance statement extracted.\n"

        # -----------------------------------------------------------------
        # B. Core Patentability Themes (detailed view)
        # -----------------------------------------------------------------
        if data.get('patent_themes'):
            md += "\n### B. Core Patentability Themes\n\n"
            md += ("The following themes represent the overarching conceptual arguments "
                   "the applicant used to establish patentability:\n\n")

            for i, theme in enumerate(data['patent_themes'], 1):
                md += f"**{i}. {theme['title']}**\n\n"
                md += f"{theme['summary']}\n\n"

                if theme.get('metaphors_or_analogies'):
                    metaphors = [m for m in theme['metaphors_or_analogies'] if m]
                    if metaphors:
                        md += f"*Key Characterizations:* {'; '.join(metaphors)}\n\n"

                if theme.get('prior_art_distinguished'):
                    arts = [a for a in theme['prior_art_distinguished'] if a]
                    if arts:
                        md += f"*Distinguishes:* {', '.join(arts)}\n\n"

                if theme.get('affected_claims'):
                    md += f"*Affected Claims:* {theme['affected_claims']}\n\n"

                if theme.get('estoppel_significance'):
                    md += f"> ⚠️ **Estoppel Significance:** {theme['estoppel_significance']}\n\n"

        # -----------------------------------------------------------------
        # C. Estoppel Events (detailed, grouped by prior art)
        # -----------------------------------------------------------------
        md += "\n### C. Estoppel Events\n"
        
        if data['estoppel_events']:
            # =====================================================================
            # Improvement D: Group estoppel events by Prior Art reference + concept
            # instead of repeating near-identical entries per claim.
            # =====================================================================
            
            # Reference name normalization map
            def _normalize_ref_name(name: str) -> str:
                if not name:
                    return name
                corrections = {
                    'maudlin': 'Mauldin', 'mauldin': 'Mauldin',
                }
                lower = name.strip().lower()
                for wrong, right in corrections.items():
                    if wrong in lower:
                        return right
                return name.strip()
            
            # Pre-build app->issued lookup
            app_to_issued = {}
            for c in data.get('claims', []):
                if c.get('mapped_final_claim') is not None:
                    app_to_issued[c['application_number']] = c['mapped_final_claim']
            
            # Step 1: Group events by normalized prior art reference
            art_event_groups = defaultdict(list)  # art_ref -> list of events
            no_art_events = []
            
            seen_texts = set()
            for event in data['estoppel_events']:
                # Normalize prior art
                if event.get('prior_art'):
                    event['prior_art'] = [_normalize_ref_name(a) for a in event['prior_art'] if a]
                
                # Dedup by text prefix
                text_key = (event.get('text') or '')[:250].strip().lower()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
                
                # Map app claims to issued claims
                affected_app_claims = event.get('claims') or []
                issued_claims = set()
                for ac in affected_app_claims:
                    if ac in app_to_issued:
                        issued_claims.add(app_to_issued[ac])
                event['issued_claims'] = sorted(issued_claims, key=_int_sort_key)
                
                # Group by prior art reference
                art_refs = [a for a in (event.get('prior_art') or []) if a and a.lower() not in ('none', 'null', 'n/a', '')]
                if art_refs:
                    for art in art_refs:
                        art_event_groups[art].append(event)
                else:
                    no_art_events.append(event)
            
            # Step 2: Within each art group, cluster by similar argument text
            from difflib import SequenceMatcher
            
            def _cluster_events(events_list):
                """Cluster events with similar text into consolidated entries."""
                clusters = []
                used = set()
                
                for i, ev_a in enumerate(events_list):
                    if i in used:
                        continue
                    cluster = [ev_a]
                    used.add(i)
                    text_a = (ev_a.get('text') or '')[:200].lower()
                    
                    for j, ev_b in enumerate(events_list):
                        if j in used:
                            continue
                        text_b = (ev_b.get('text') or '')[:200].lower()
                        sim = SequenceMatcher(None, text_a, text_b).ratio()
                        if sim >= 0.6:
                            cluster.append(ev_b)
                            used.add(j)
                    
                    clusters.append(cluster)
                return clusters
            
            # Step 3: Render grouped by prior art
            for art_ref in sorted(art_event_groups.keys()):
                events_for_art = art_event_groups[art_ref]
                clusters = _cluster_events(events_for_art)
                
                # Collect all issued claims affected by this art
                all_issued = set()
                for ev in events_for_art:
                    all_issued.update(ev.get('issued_claims', []))
                
                claims_label = f" (Affecting Issued Claims: {sorted(all_issued, key=_int_sort_key)})" if all_issued else ""
                md += f"\n**Arguments distinguishing {art_ref}**{claims_label}\n"
                
                for cluster in clusters:
                    # Use the longest/best event text as representative
                    representative = max(cluster, key=lambda e: len(e.get('text', '')))
                    risk = representative.get('risk_level', 'unknown')
                    element = representative.get('element', '')
                    element_label = f" — *{element}*" if element and element.lower() not in ('none', 'general', 'n/a', '') else ""
                    
                    raw_text = representative.get('text', '')
                    art_refs_for_reg = [a for a in (representative.get('prior_art') or []) if a and a.lower() not in ('none', 'null', 'n/a', '')]
                    
                    # v2.2 Workstream 1: Check if this argument was already rendered elsewhere
                    already_in = registry.is_rendered(raw_text, art_refs_for_reg)
                    if already_in:
                        md += f"\n- **{risk.capitalize()} risk**{element_label}\n"
                        md += f"  - *(See {already_in} above for full argument)*\n"
                        if representative.get('recapture_analysis'):
                            md += f"  - ⚠️ *{representative['recapture_analysis']}*\n"
                        continue
                    
                    # Increased limits significantly to prevent truncation
                    text_limit = 5000
                    text = raw_text[:text_limit]
                    if len(raw_text) > text_limit:
                        last_period = text.rfind('. ')
                        last_semicolon = text.rfind('; ')
                        boundary = max(last_period, last_semicolon)
                        if boundary > text_limit * 0.5:
                            text = text[:boundary + 1]
                    suffix = '...' if len(raw_text) > len(text) else ''
                    
                    md += f"\n- **{risk.capitalize()} risk**{element_label}\n"
                    md += f"  - {text}{suffix}\n"
                    
                    if representative.get('recapture_analysis'):
                        md += f"  - ⚠️ *{representative['recapture_analysis']}*\n"
                    
                    # Register this argument
                    registry.register(raw_text, "Estoppel Events", art_refs_for_reg)
            
            # Render events with no prior art
            if no_art_events:
                md += f"\n**General Estoppel Events (no specific prior art cited):**\n"
                clusters = _cluster_events(no_art_events)
                for cluster in clusters:
                    representative = max(cluster, key=lambda e: len(e.get('text', '')))
                    risk = representative.get('risk_level', 'unknown')
                    element = representative.get('element', '')
                    element_label = f" — *{element}*" if element and element.lower() not in ('none', 'general', 'n/a', '') else ""
                    
                    all_issued = set()
                    for ev in cluster:
                        all_issued.update(ev.get('issued_claims', []))
                    claims_note = f" [Claims: {sorted(all_issued, key=_int_sort_key)}]" if all_issued else ""
                    
                    raw_text = representative.get('text', '')
                    
                    # v2.2: Check if already rendered
                    already_in = registry.is_rendered(raw_text)
                    if already_in:
                        md += f"\n- **{risk.capitalize()} risk**{element_label}{claims_note}\n"
                        md += f"  - *(See {already_in} above for full argument)*\n"
                        continue
                    
                    # Increased limits significantly
                    text_limit = 5000
                    text = raw_text[:text_limit]
                    if len(raw_text) > text_limit:
                        last_period = text.rfind('. ')
                        last_semicolon = text.rfind('; ')
                        boundary = max(last_period, last_semicolon)
                        if boundary > text_limit * 0.5:
                            text = text[:boundary + 1]
                    suffix = '...' if len(raw_text) > len(text) else ''
                    
                    md += f"\n- **{risk.capitalize()} risk**{element_label}{claims_note}\n"
                    md += f"  - {text}{suffix}\n"
                    if representative.get('recapture_analysis'):
                        md += f"  - ⚠️ *{representative['recapture_analysis']}*\n"
                    
                    registry.register(raw_text, "Estoppel Events")
        else:
            md += "\nNo significant estoppel events identified.\n"
        
        # Terminal Disclaimers & Priority Flags are now in Strategic Summary;
        # Validity Risks (Shadow Examiner) are integrated per-family above.
        # Keep Restriction Requirements as a detail section.

        # -----------------------------------------------------------------
        # D. Restriction Requirements
        # -----------------------------------------------------------------
        md += "\n### D. Restriction Requirements\n"
        if data['restrictions']:
            for restr in data['restrictions']:
                md += f"\n- Date: {restr.get('date', 'N/A')}\n"
                md += f"  - Elected Group: {restr.get('elected_group', 'N/A')}\n"
                md += f"  - Non-Elected Claims: {restr.get('non_elected_claims', 'N/A')}\n"
                md += f"  - Traversed: {'Yes' if restr.get('traversed') else 'No'}\n"
        else:
            md += "\nNo restriction requirements found.\n"
        
        # (Reasons for Allowance rendered above in Executive Summary)
        
        # =====================================================================
        # E. Key Claim Limitations & Prosecution Positions
        # =====================================================================
        md += "\n---\n\n### E. Key Claim Limitations & Prosecution Positions\n\n"
        md += "This section aggregates the substantive limitations added during prosecution, "
        md += "the prior art they were designed to overcome, and their estoppel significance.\n"
        
        # Build limitations from added_limitations across all claim versions
        key_limitations = []
        for claim_data in data['claims']:
            for v in claim_data.get('versions', []):
                if v.get('version', 1) == 1:
                    continue  # Skip original filing
                for lim in (v.get('added_limitations') or []):
                    lim_text = lim.get('text', '')
                    if not lim_text or len(lim_text.strip()) < 5:
                        continue
                    key_limitations.append({
                        'claim': claim_data['application_number'],
                        'issued': claim_data.get('mapped_final_claim'),
                        'text': lim_text,
                        'art': lim.get('likely_response_to_art', ''),
                        'argument': lim.get('applicant_argument_summary', ''),
                        'date': v.get('date'),
                    })
        
        if key_limitations:
            # Fix #15: Sort by issued claim number for easier reference
            key_limitations.sort(key=lambda lim: (_int_sort_key(lim.get('issued') or 999), _int_sort_key(lim.get('claim', 0))))
            
            md += "\n| App Claim | Issued As | Limitation Added | Prior Art | Applicant's Argument |\n"
            md += "|-----------|-----------|------------------|-----------|---------------------|\n"
            for lim in key_limitations:
                claim_label = f"{lim['claim']}"
                issued_label = str(lim.get('issued') or '-')
                # Increased table cell limits
                text = (lim.get('text') or '').replace('|', ' ').replace('\n', ' ')[:500]
                art = (lim.get('art') or '-').replace('|', ' ').replace('\n', ' ')
                arg = (lim.get('argument') or '-').replace('|', ' ').replace('\n', ' ')[:500]
                md += f"| {claim_label} | {issued_label} | \"{text}\" | {art} | {arg} |\n"
        else:
            md += "\nNo substantive claim limitations tracked.\n"
        
        # =====================================================================
        # Fix #3: Key Prosecution Positions — synthesize thematic distinctions,
        # directionality constraints, and purpose characterizations into
        # clear, standalone positions separate from the limitations table.
        # =====================================================================
        md += "\n### Key Prosecution Positions\n\n"
        md += "These are high-level conceptual positions taken during prosecution "
        md += "that constrain claim scope beyond specific textual amendments.\n"
        
        prosecution_positions = []
        for stmt in data['statements']:
            ctx = stmt.get('context') or ''
            text = stmt.get('text') or ''
            # Collect thematic distinctions, directionality constraints, and purpose characterizations
            is_thematic = (
                '[DIRECTIONALITY CONSTRAINT]' in ctx
                or '[PURPOSE/FUNCTION CHARACTERIZATION]' in ctx
                or 'fundamentally different' in text.lower()
                or 'cannot be used' in text.lower()
                or 'does not include' in text.lower()
                or 'is not directed to' in text.lower()
                or 'serves as a' in text.lower()
            )
            if is_thematic and len(text) > 20:
                position_type = "Directionality Constraint" if '[DIRECTIONALITY CONSTRAINT]' in ctx \
                    else "Purpose/Function" if '[PURPOSE/FUNCTION CHARACTERIZATION]' in ctx \
                    else "Categorical Exclusion" if any(p in text.lower() for p in ['cannot be used', 'does not include']) \
                    else "Conceptual Distinction"
                prosecution_positions.append({
                    'type': position_type,
                    'text': text,
                    'art': ', '.join(stmt.get('prior_art') or []) if stmt.get('prior_art') else '-',
                    'claims': stmt.get('affected_claims'),
                    'element': stmt.get('element_defined'),
                })
        
        if prosecution_positions:
            for pos in prosecution_positions:
                claims_str = f" (Claims: {pos['claims']})" if pos.get('claims') else ""
                element_str = f" — **{pos['element']}**" if pos.get('element') else ""
                md += f"\n**{pos['type']}**{element_str}{claims_str}\n"
                # Removed truncation
                md += f"> {pos['text']}\n"
                if pos['art'] != '-':
                    md += f"> *Distinguishing:* {pos['art']}\n"
                md += "\n"
        else:
            md += "\nNo high-level prosecution positions identified.\n"
        
        # =====================================================================
        # Gap 6: Categorical Exclusions — Dedicated Sub-Section
        # =====================================================================
        if data.get('categorical_exclusions'):
            md += "\n### Categorical Exclusions\n\n"
            md += "These are hard constraints on what the claims can cover, based on "
            md += "negative limitations and explicit disavowals during prosecution.\n\n"
            seen_excl = set()
            for excl in data['categorical_exclusions']:
                text_key = excl['text'][:80]
                if text_key in seen_excl:
                    continue
                seen_excl.add(text_key)
                claims_str = f" (Claims: {excl['claims']})" if excl.get('claims') else ""
                art_str = f"; distinguishing {', '.join(excl['prior_art'])}" if excl.get('prior_art') else ""
                element_str = f" — **{excl['element']}**" if excl.get('element') else ""
                md += f"- ❌{element_str}{claims_str}{art_str}\n"
                md += f"  > {excl['text']}\n\n"  # Gap 11: Never truncate exclusion text
        
        # =====================================================================
        # Gap 5: Directionality Constraints — Dedicated Sub-Section
        # =====================================================================
        if data.get('directionality_constraints'):
            md += "\n### Directionality Constraints\n\n"
            md += "These specify required directions (A→B, not B→A) that are critical "
            md += "for infringement analysis.\n\n"
            seen_dir = set()
            for dc in data['directionality_constraints']:
                text_key = dc['text'][:80]
                if text_key in seen_dir:
                    continue
                seen_dir.add(text_key)
                claims_str = f" (Claims: {dc['claims']})" if dc.get('claims') else ""
                art_str = f" *Distinguishing: {', '.join(dc['prior_art'])}*" if dc.get('prior_art') else ""
                md += f"- ↔️ **{dc.get('element', 'Direction')}**{claims_str}\n"
                md += f"  > {dc['text']}\n"  # Gap 11: Never truncate directionality text
                if art_str:
                    md += f"  > {art_str}\n"
                md += "\n"
        
        # =====================================================================
        # Gap 7: Purpose/Function Characterizations — Dedicated Sub-Section
        # =====================================================================
        if data.get('purpose_characterizations'):
            md += "\n### Purpose/Function Characterizations\n\n"
            md += "The applicant characterized the purpose or function of the invention "
            md += "in ways that constrain claim scope.\n\n"
            seen_purp = set()
            for pc in data['purpose_characterizations']:
                text_key = pc['text'][:80]
                if text_key in seen_purp:
                    continue
                seen_purp.add(text_key)
                claims_str = f" (Claims: {pc['claims']})" if pc.get('claims') else ""
                md += f"- 🎯 **{pc.get('element', 'Purpose')}**{claims_str}\n"
                md += f"  > {pc['text']}\n\n"  # Gap 11: Never truncate scope-constraining text
        
        # =====================================================================
        # Gap 1: Strategic Tensions / Contradictions — Dedicated Sub-Section
        # =====================================================================
        if data.get('strategic_tensions'):
            md += "\n### ⚠️ Strategic Tensions / Contradictions\n\n"
            md += "The applicant argued seemingly contradictory positions at different points "
            md += "in prosecution. A litigator may exploit these.\n\n"
            for tension in data['strategic_tensions']:
                text = tension.get('text', '')
                ctx = tension.get('context') or ''
                claims_str = f" (Claims: {tension.get('affected_claims')})" if tension.get('affected_claims') else ""
                md += f"**Strategic Tension**{claims_str}\n"
                # Removed truncation
                md += f"> {text}\n\n"
        
        # (Scope Differentials are now rendered in the Strategic Summary above)
        
        # (Per-Claim Scope Constraints are now integrated into the Strategic Summary per-family blocks)

        # Claim Evolution — v2.2: Event-Based with Mirror Condensation
        md += "\n---\n\n### F. Selected Claim Histories\n"
        
        # =====================================================================
        # Gap 4 fix: Prioritize claims that map to final issued claims,
        # then other independent claims. Include key dependent claims.
        # =====================================================================
        all_claims = data['claims']
        issued_claims = sorted(
            [c for c in all_claims if c.get('mapped_final_claim') is not None],
            key=lambda c: _int_sort_key(c.get('mapped_final_claim', 999))
        )
        other_independent = sorted(
            [c for c in all_claims
             if c.get('is_independent')
             and c.get('mapped_final_claim') is None
             and c.get('status') in ('Allowed', 'Pending', 'Rejected')],
            key=lambda c: _int_sort_key(c['application_number'])
        )
        # Also include dependent claims that are strategically significant:
        key_dependents = sorted(
            [c for c in all_claims
             if not c.get('is_independent')
             and c.get('mapped_final_claim') is not None
             and (
                 len(c.get('versions', [])) >= 2
                 or any(
                     lim.get('likely_response_to_art', '').lower() not in ('', 'none', 'n/a', 'not specified')
                     for v in c.get('versions', [])
                     for lim in (v.get('added_limitations') or [])
                 )
             )],
            key=lambda c: _int_sort_key(c.get('mapped_final_claim', 999))
        )
        
        claims_to_show = issued_claims + other_independent + key_dependents
        # Deduplicate
        seen = set()
        deduped = []
        for c in claims_to_show:
            if c['application_number'] not in seen:
                seen.add(c['application_number'])
                deduped.append(c)
        all_issued_app_nums = {c['application_number'] for c in issued_claims}
        for c in data['claims']:
            if c.get('mapped_final_claim') is not None and c['application_number'] not in seen:
                seen.add(c['application_number'])
                deduped.append(c)
        claims_to_show = deduped[:20]
        
        # =====================================================================
        # v2.2 Workstream 6: Mirror Claim Detection
        # =====================================================================
        families_for_mirror = self._group_independent_claims(
            data.get('final_claims', []),
            data.get('scope_differentials', []),
            data.get('claims', []),
        )
        mirror_map = self._detect_mirror_claims(families_for_mirror, data['claims'])
        
        for claim in claims_to_show:
            final_num = claim.get('mapped_final_claim')
            primary = mirror_map.get(final_num) if final_num is not None else None
            
            md += f"\n### Claim {claim['application_number']}"
            if claim.get('issued_number'):
                md += f" → Issued Claim {claim['issued_number']}"
            if claim.get('mapped_final_claim'):
                md += f" (Mapped to Final Claim {claim['mapped_final_claim']})"
            md += "\n\n"
            
            # =====================================================================
            # v2.2 Workstream 6: Condensed rendering for mirror claims
            # =====================================================================
            if primary is not None:
                # Find the primary claim's type hint
                primary_claim = next(
                    (c for c in claims_to_show if c.get('mapped_final_claim') == primary),
                    None
                )
                type_hint = ""
                if primary_claim:
                    fc = next((f for f in data['final_claims'] if f['number'] == final_num), None)
                    if fc:
                        fc_text = (fc.get('text') or '').lower()
                        if 'method' in fc_text[:80] or 'process' in fc_text[:80]:
                            type_hint = " (Method variant)"
                        elif 'medium' in fc_text[:120] or 'computer-readable' in fc_text[:120]:
                            type_hint = " (Computer-Readable Medium variant)"
                        elif 'system' in fc_text[:80] or 'apparatus' in fc_text[:80]:
                            type_hint = " (System variant)"
                
                md += (
                    f"This claim mirrors Claim {primary}'s prosecution history{type_hint}. "
                    f"It underwent the same amendments on the same dates "
                    f"for the same reasons.\n\n"
                )
                md += "See Appendix A for full version texts.\n\n"
                continue
            
            # =====================================================================
            # Full rendering for primary claims — event-based (v2.2 Workstream 3)
            # =====================================================================
            
            # Claim narrative summary (Layer 2 synthesis)
            narrative = data.get('claim_narratives', {}).get(claim['application_number'])
            if narrative:
                md += f"**Narrative:** {narrative['evolution_summary']}\n\n"
                if narrative.get('turning_point_event'):
                    md += f"> 🔑 **Turning Point:** {narrative['turning_point_event']}\n\n"
            
            # Use event-based renderer instead of version-by-version
            md += f"#### Prosecution History of Claim {claim['application_number']}\n\n"
            md += self._render_claim_evolution_event_based(claim, data, registry)
            
            # Inject Reasons for Allowance relevant to this claim
            app_num = claim['application_number']
            if data.get('reasons_for_allowance'):
                relevant_rfa = [
                    rfa for rfa in data['reasons_for_allowance']
                    if app_num in (rfa.get('affected_claims') or [])
                ]
                if relevant_rfa:
                    md += "**Examiner's Reasons for Allowance (this claim):**\n\n"
                    for rfa in relevant_rfa:
                        md += f"> {rfa.get('text', 'N/A')}\n\n"
        
        # Prior Art Distinctions (Gap 1 fix: unified from statements + added_limitations)
        md += "\n---\n\n### G. Prior Art Distinctions\n"
        
        # Gap 4: Consolidated Prior Art Reference Summary
        prior_art_refs = data.get('prior_art_refs', [])
        if prior_art_refs:
            md += "\n**Consolidated Prior Art References:**\n\n"
            md += "| Reference | Basis | Claims | Overcome? | Key Teachings |\n"
            md += "|-----------|-------|--------|-----------|---------------|\n"
            for ref in prior_art_refs:
                name = ref.get('canonical_name', 'Unknown')
                pub = ref.get('patent_or_pub_number', '')
                if pub:
                    name += f" ({pub})"
                basis = ref.get('applied_basis', '-')
                claims = ', '.join(str(c) for c in (ref.get('affected_claims') or []))[:40] or '-'
                overcome = '✅ Yes' if ref.get('is_overcome') else '❌ No'
                teachings = '; '.join((ref.get('key_teachings') or [])[:2])[:80] or '-'
                md += f"| {name} | {basis} | {claims} | {overcome} | {teachings} |\n"
            md += "\n"
        
        prior_art_map = data.get('prior_art_map', {})
        
        # Also gather from statements as a fallback (for backward compatibility)
        if not prior_art_map:
            distinctions = [s for s in data['statements'] 
                           if s.get('prior_art')]
            for dist in distinctions:
                for art_ref in (dist.get('prior_art') or []):
                    if art_ref and art_ref.lower() not in ('none', 'null', 'n/a', ''):
                        if art_ref not in prior_art_map:
                            prior_art_map[art_ref] = []
                        prior_art_map[art_ref].append({
                            'text': dist.get('text', ''),
                            'speaker': dist.get('speaker', 'Applicant'),
                            'claims': dist.get('affected_claims'),
                            'element': dist.get('element_defined'),
                            'category': dist.get('category', ''),
                        })
        
        if prior_art_map:
            for art_ref, entries in sorted(prior_art_map.items()):
                md += f"\n### {art_ref}\n\n"
                
                # Collect all affected claims across entries
                all_claims_for_art = set()
                for entry in entries:
                    for c in (entry.get('claims') or []):
                        all_claims_for_art.add(c)
                
                if all_claims_for_art:
                    md += f"**Affected Claims:** {sorted(all_claims_for_art, key=_int_sort_key)}\n\n"
                
                for entry in entries:
                    speaker = entry.get('speaker', 'Applicant')
                    text = entry.get('text', 'N/A')
                    argument = entry.get('argument', '')
                    
                    # v2.2 Workstream 1: Check if this argument was already rendered
                    full_text = argument or text
                    art_refs_for_reg = [art_ref]
                    already_in = registry.is_rendered(full_text, art_refs_for_reg)
                    
                    if already_in:
                        md += f"- *(See {already_in} for full argument)*\n\n"
                        continue
                    
                    if argument:
                        md += f"**{speaker} (Amendment):** Added \"{text}\"\n"
                        md += f"> *Argument:* {argument}\n\n"
                    else:
                        md += f"**{speaker}:**\n"
                        md += f"> {text}\n\n"
                    
                    registry.register(full_text, f"Prior Art: {art_ref}", art_refs_for_reg)
        else:
            md += "\nNo prior art distinctions extracted.\n"
        
        # =====================================================================
        # H. Claim Genealogy Map
        # =====================================================================
        md += "\n---\n\n### H. Claim Genealogy Map\n"
        
        genealogy = data['genealogy']
        
        md += "\n### Application → Issued Claim Mapping\n\n"
        
        if data['final_claims']:
            md += "| App Claim | Issued Claim | Final Claim Match | Confidence | Status |\n"
            md += "|-----------|--------------|-------------------|------------|--------|\n"
        else:
            md += "| Application Claim | Issued Claim | Status |\n"
            md += "|-------------------|--------------|--------|\n"
        
        for claim_data in sorted(data['claims'], key=lambda x: _int_sort_key(x['application_number'])):
            app_num = claim_data['application_number']
            issued_num = claim_data.get('issued_number') or '-'
            status = claim_data['status']
            
            if data['final_claims']:
                mapped = claim_data.get('mapped_final_claim') or '-'
                confidence = f"{claim_data.get('mapping_confidence', 0)*100:.0f}%" if claim_data.get('mapping_confidence') else '-'
                md += f"| {app_num} | {issued_num} | {mapped} | {confidence} | {status} |\n"
            else:
                md += f"| {app_num} | {issued_num} | {status} |\n"
        
        md += "\n### Cancelled Claims\n"
        md += f"{genealogy.get('cancelled', []) or 'None'}\n"
        
        md += "\n### Withdrawn (Non-Elected) Claims\n"
        md += f"{genealogy.get('withdrawn', []) or 'None'}\n"
        
        # Means-Plus-Function Alert
        md += "\n### Means-Plus-Function Elements (112(f))\n"
        mpf_claims = [c for c in data['claims'] if c.get('is_means_plus_function')]
        if mpf_claims:
            for claim in mpf_claims:
                md += f"\n- **Claim {claim['application_number']}**\n"
                if claim.get('mpf_elements'):
                    for mpf in claim['mpf_elements']:
                        md += f"  - Element: {mpf.get('element_text', 'N/A')}\n"
                        md += f"  - Structure: {mpf.get('corresponding_structure', 'N/A')}\n"
        else:
            md += "\nNo means-plus-function elements identified.\n"
        
        # Prosecution Timeline — v2.2: Brief summary only, full timeline in Appendix D
        md += "\n---\n\n### I. Prosecution Timeline (Summary)\n\n"
        
        docs_with_idx = [(i, doc) for i, doc in enumerate(data['documents'])]
        docs_sorted = sorted(
            docs_with_idx,
            key=lambda x: (
                x[1].get('date') or '9999-99-99',
                x[0]
            )
        )
        
        # Show only high-priority documents in the main body
        high_priority_docs = [(i, d) for i, d in docs_sorted if d.get('is_high_priority')]
        if high_priority_docs:
            md += "**Key Prosecution Events:**\n\n"
            for _idx, doc in high_priority_docs:
                date = doc.get('date', 'Unknown')
                if date and len(date) >= 10:
                    date = date[:10]
                elif not date:
                    date = 'Unknown'
                doc_type = doc.get('type', 'Unknown')
                pages = f" (pp. {doc['pages']})" if doc.get('pages') else ""
                md += f"- **{date}**: ⭐ {doc_type}{pages}\n"
            md += f"\n*See Appendix D for the complete prosecution timeline ({len(docs_sorted)} documents).*\n"
        else:
            md += f"*See Appendix D for the complete prosecution timeline ({len(docs_sorted)} documents).*\n"
        
        # Gap 9: Prosecution Milestones
        milestones = data.get('milestones', [])
        if milestones:
            md += "\n**Prosecution Milestones:**\n\n"
            for m in milestones:
                date_str = m.get('date', 'Unknown')
                if date_str and len(date_str) >= 10:
                    date_str = date_str[:10]
                m_type = m.get('type', 'Unknown')
                emoji = {'RCE': '🔄', 'Appeal': '⚖️', 'Continuation': '🔗', 'Petition': '📋'}.get(m_type, '📌')
                context = f" — {m['context']}" if m.get('context') else ""
                md += f"- **{date_str}**: {emoji} {m_type}{context}\n"
            md += "\n"
        
        # ═══════════════════════════════════════════════════════════════
        # APPENDIX (v2.2 — raw evidence locker)
        # ═══════════════════════════════════════════════════════════════
        md += "\n\n---\n\n"
        md += "# Appendix\n\n"
        md += (
            "*The following sections contain the complete raw data "
            "underlying the analysis above. They are provided for "
            "reference and verification.*\n\n"
        )
        
        # ── Appendix A: Full Claim Version Texts ──
        md += "## Appendix A: Full Claim Version Texts\n\n"
        for claim in sorted(data['claims'], key=lambda c: _int_sort_key(c['application_number'])):
            md += f"### Claim {claim['application_number']}"
            if claim.get('issued_number'):
                md += f" (Issued as Claim {claim['issued_number']})"
            md += "\n\n"
            for v in claim.get('versions', []):
                date = (v.get('date') or 'N/A')[:10]
                md += f"**Version {v['version']}** ({date})\n\n"
                text = v.get('text', 'N/A')
                md += f"```\n{text}\n```\n\n"
        
        # ── Appendix B: Complete Lexicon (Unfiltered) ──
        md += "## Appendix B: Complete Lexicon (Unfiltered)\n\n"
        md += "*All claim element definitions, including those filtered from the main report.*\n\n"
        if data['lexicon']:
            for element, definitions in data['lexicon'].items():
                md += f"### {element}\n\n"
                narrative = data.get('term_narratives', {}).get(element)
                if narrative:
                    md += f"**Synthesis:** {narrative['summary']}\n\n"
                    status = narrative.get('status', 'Unknown')
                    if status == 'Contradictory':
                        md += f"> ⚠️ **Inconsistency Detected:** {narrative.get('contradiction_details', '')}\n\n"
                    elif status == 'Evolving':
                        md += f"> ℹ️ **Definition Evolved:** {narrative.get('contradiction_details', '')}\n\n"
                for defn in definitions:
                    speaker = defn.get('speaker', 'Unknown')
                    acquiesced = " ✓ (Acquiesced)" if defn.get('is_acquiesced') else ""
                    md += f"**{speaker}**{acquiesced}:\n"
                    md += f"> {defn.get('definition', 'N/A')}\n\n"
        else:
            md += "No claim term definitions extracted.\n\n"
        
        # ── Appendix C: All Prosecution Statements ──
        md += "## Appendix C: All Prosecution Statements\n\n"
        md += "*Complete dump of all extracted prosecution statements, organized chronologically.*\n\n"
        if data['statements']:
            for stmt in data['statements']:
                speaker = stmt.get('speaker', 'Unknown')
                category = stmt.get('category', 'General')
                text = stmt.get('text', 'N/A')
                claims = stmt.get('affected_claims', [])
                art = stmt.get('prior_art', [])
                
                md += f"**{speaker}** ({category})"
                if claims:
                    md += f" — Claims: {claims}"
                if art:
                    art_str = ', '.join(a for a in art if a and a.lower() not in ('none', 'null', 'n/a', ''))
                    if art_str:
                        md += f" — Art: {art_str}"
                md += "\n"
                md += f"> {text}\n\n"
        else:
            md += "No prosecution statements extracted.\n\n"
        
        # ── Appendix D: Prosecution Timeline ──
        md += "## Appendix D: Prosecution Timeline\n\n"
        for _idx, doc in docs_sorted:
            date = doc.get('date', 'Unknown')
            if date and len(date) >= 10:
                date = date[:10]
            elif not date:
                date = 'Unknown'
            doc_type = doc.get('type', 'Unknown')
            priority = " ⭐" if doc.get('is_high_priority') else ""
            pages = f" (pp. {doc['pages']})" if doc.get('pages') else ""
            md += f"- **{date}**: {doc_type}{priority}{pages}\n"
        
        # ── Appendix E: Prior Art Distinctions (Complete) ──
        md += "\n## Appendix E: Prior Art Distinctions (Complete)\n\n"
        md += "*Full prior art distinctions map, unfiltered and un-deduplicated.*\n\n"
        prior_art_map_full = data.get('prior_art_map', {})
        if prior_art_map_full:
            for art_ref, entries in sorted(prior_art_map_full.items()):
                md += f"### {art_ref}\n\n"
                for entry in entries:
                    speaker = entry.get('speaker', 'Applicant')
                    text = entry.get('text', 'N/A')
                    argument = entry.get('argument', '')
                    md += f"**{speaker}:**\n"
                    md += f"> {text}\n"
                    if argument:
                        md += f"> *Argument:* {argument}\n"
                    md += "\n"
        else:
            md += "No prior art distinctions extracted.\n\n"
        
        return md
    
    def _generate_html(self, data: Dict) -> str:
        """Generate HTML report"""
        patent = data['patent']
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File History Review Report - {patent.get('patent_number') or patent.get('application_number') or 'Analysis'}</title>
    <style>
        :root {{
            --primary-color: #1e3a5f;
            --accent-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --bg-color: #f8f9fa;
            --text-color: #2c3e50;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            margin: 0;
            padding: 0;
            background-color: #fff;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }}
        
        h1 {{ margin: 0 0 1rem 0; }}
        h2 {{
            color: var(--primary-color);
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 0.5rem;
        }}
        h3 {{ color: var(--primary-color); }}
        
        .meta {{ opacity: 0.9; }}
        
        .section {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .final-claims {{
            background: linear-gradient(135deg, #e8f4f8, #f0f7fa);
            border-left: 4px solid var(--accent-color);
        }}
        
        .final-claim {{
            background: white;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
            border-left: 3px solid var(--success-color);
        }}
        
        .final-claim h4 {{
            color: var(--success-color);
            margin-top: 0;
        }}
        
        .alert {{
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }}
        
        .alert-warning {{ background-color: #fef3e2; border-left: 4px solid var(--warning-color); }}
        .alert-danger {{ background-color: #fce4e4; border-left: 4px solid var(--danger-color); }}
        .alert-success {{ background-color: #e8f5e9; border-left: 4px solid var(--success-color); }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        
        .badge-high {{ background-color: var(--danger-color); color: white; }}
        .badge-medium {{ background-color: var(--warning-color); color: white; }}
        .badge-low {{ background-color: var(--success-color); color: white; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background-color: var(--bg-color);
            font-weight: 600;
        }}
        
        .claim-text {{
            background-color: var(--bg-color);
            padding: 1rem;
            border-radius: 4px;
            font-family: monospace;
            white-space: pre-wrap;
            font-size: 0.9rem;
            margin: 0.5rem 0;
        }}
        
        .timeline {{
            position: relative;
            padding-left: 2rem;
        }}
        
        .timeline-item {{
            padding: 0.5rem 0 0.5rem 1rem;
            border-left: 2px solid var(--accent-color);
            position: relative;
        }}
        
        .timeline-item::before {{
            content: '';
            position: absolute;
            left: -6px;
            top: 50%;
            transform: translateY(-50%);
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: var(--accent-color);
        }}
        
        .timeline-item.high-priority::before {{
            background-color: var(--warning-color);
        }}
        
        blockquote {{
            border-left: 3px solid var(--accent-color);
            padding-left: 1rem;
            margin: 0.5rem 0;
            font-style: italic;
            color: #4a5568;
        }}
        
        .lexicon-entry {{
            margin-bottom: 1.5rem;
            padding: 1rem;
            background-color: var(--bg-color);
            border-radius: 6px;
        }}
        
        .lexicon-entry h4 {{
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }}
        
        .acquiesced {{
            color: var(--success-color);
            font-weight: 500;
        }}
        
        .ground-truth-badge {{
            background: var(--success-color);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.9rem;
            margin-left: 0.5rem;
        }}
        
        .synthesis-box {{
            background: linear-gradient(135deg, #f0f4ff, #e8ecf8);
            border-left: 4px solid var(--accent-color);
            padding: 1rem 1.25rem;
            border-radius: 4px;
            margin: 0.75rem 0;
            font-style: italic;
        }}
        
        .synthesis-box strong {{
            font-style: normal;
        }}
        
        .consistency-warning {{
            background-color: #fef3e2;
            border-left: 4px solid var(--warning-color);
            padding: 0.75rem 1rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        }}
        
        .consistency-ok {{
            background-color: #e8f5e9;
            border-left: 4px solid var(--success-color);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            margin: 0.5rem 0;
            font-size: 0.9rem;
        }}
        
        .narrative-box {{
            background: #f7f9fc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 1rem 1.25rem;
            margin: 0.75rem 0;
        }}
        
        .narrative-box .turning-point {{
            color: var(--primary-color);
            font-weight: 600;
            margin-top: 0.5rem;
        }}
        
        .risk-card {{
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 1rem;
            margin: 0.75rem 0;
        }}
        
        .risk-high {{ border-left: 4px solid var(--danger-color); }}
        .risk-medium {{ border-left: 4px solid var(--warning-color); }}
        .risk-low {{ border-left: 4px solid var(--success-color); }}
        
        .raw-evidence-toggle {{
            cursor: pointer;
            color: var(--accent-color);
            font-size: 0.9rem;
            margin-top: 0.5rem;
            text-decoration: underline;
        }}
        
        .raw-evidence {{
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px dashed #ddd;
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: #718096;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>File History Review Report</h1>
            <div class="meta">
                <p><strong>Patent/Application:</strong> {patent.get('patent_number') or patent.get('application_number') or 'N/A'}</p>
                <p><strong>Title:</strong> {patent.get('title') or 'N/A'}</p>
                <p><strong>Filing Date:</strong> {patent.get('filing_date') or 'N/A'}</p>
                <p><strong>Report Generated:</strong> {data['generated_at']}</p>
            </div>
        </header>
"""
        
        # Executive Summary
        html += """
        <div class="section">
            <h2>A. Executive Summary</h2>
            
            <h3>Estoppel Events</h3>
"""
        
        if data['estoppel_events']:
            for event in data['estoppel_events'][:10]:
                risk = event.get('risk_level', 'unknown')
                badge_class = f"badge-{risk}" if risk in ['high', 'medium', 'low'] else 'badge-medium'
                # Removed truncation
                html += f"""
            <div class="alert alert-warning">
                <strong>{event.get('element', 'General')}</strong> 
                <span class="badge {badge_class}">{risk.upper()}</span>
                <p>Claims: {event.get('claims', 'N/A')}</p>
                <blockquote>{event.get('text', '')}</blockquote>
            </div>
"""
        else:
            html += '<div class="alert alert-success">No significant estoppel events identified.</div>'
        
        # Terminal Disclaimers
        html += """
            <h3>Terminal Disclaimers</h3>
"""
        if data['terminal_disclaimers']:
            for td in data['terminal_disclaimers']:
                html += f"""
            <div class="alert alert-danger">
                <strong>Disclaimed over:</strong> {td.get('disclaimed_patent', 'N/A')}<br>
                <strong>Date:</strong> {td.get('date', 'N/A')}<br>
                <strong>Reason:</strong> {td.get('reason', 'N/A')}
            </div>
"""
        else:
            html += '<p>No terminal disclaimers found.</p>'
        
        # Validity Risks (Shadow Examiner)
        html += """
            <h3>Validity Risk Analysis <span class="badge badge-medium">Shadow Examiner</span></h3>
"""
        if data.get('validity_risks'):
            for risk in data['validity_risks']:
                severity = risk.get('severity', 'Medium').lower()
                html += f"""
            <div class="risk-card risk-{severity}">
                <strong>{risk.get('risk_type', 'Unknown')}</strong>
                <span class="badge badge-{severity}">{risk.get('severity', 'Unknown').upper()}</span>
                <p>{risk.get('description', 'N/A')}</p>
"""
                if risk.get('reasoning'):
                    html += f'                <p><em>Reasoning:</em> {risk["reasoning"]}</p>\n'
                if risk.get('affected_claims'):
                    html += f'                <p><em>Affected Claims:</em> {risk["affected_claims"]}</p>\n'
                html += "            </div>\n"
        else:
            html += '<div class="alert alert-success">No validity risks identified by the Shadow Examiner.</div>'
        
        # Reasons for Allowance (Gap 8 fix)
        html += """
            <h3>Reasons for Allowance</h3>
"""
        if data.get('reasons_for_allowance'):
            for rfa in data['reasons_for_allowance']:
                # Removed truncation
                html += f"""
            <div class="alert alert-success">
                <blockquote>{rfa.get('text', 'N/A')}</blockquote>
"""
                if rfa.get('affected_claims'):
                    html += f'                <p><em>Affected Claims:</em> {rfa["affected_claims"]}</p>\n'
                html += "            </div>\n"
        else:
            html += '<p>No Reasons for Allowance statement extracted.</p>'
        
        # Close Executive Summary, insert Final Claims, then start Genealogy
        html += """
        </div>
"""
        
        # Final Claims Section
        if data['final_claims']:
            html += """
        <div class="section final-claims">
            <h2>Final Issued Claims <span class="ground-truth-badge">Ground Truth</span></h2>
            <p>The following claims represent the final issued claims from the patent:</p>
"""
            for fc in data['final_claims']:
                claim_type = "Independent" if fc['is_independent'] else f"Depends on claim {fc['depends_on']}"
                html += f"""
            <div class="final-claim">
                <h4>Claim {fc['number']} ({claim_type})</h4>
                <div class="claim-text">{fc['text']}</div>
            </div>
"""
            html += "        </div>\n"
        
        html += """
        <div class="section">
            <h2>B. Claim Genealogy Map</h2>
            
            <h3>Application → Issued Claim Mapping</h3>
            <table>
                <thead>
                    <tr>
                        <th>Application Claim</th>
                        <th>Issued Claim</th>
"""
        
        if data['final_claims']:
            html += """                        <th>Final Claim Match</th>
                        <th>Confidence</th>
"""
        
        html += """                        <th>Status</th>
                        <th>Type</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for claim_data in sorted(data['claims'], key=lambda x: _int_sort_key(x['application_number'])):
            status_class = ""
            if claim_data['status'] == 'Allowed':
                status_class = "style='color: var(--success-color)'"
            elif 'Cancelled' in claim_data['status'] or 'Withdrawn' in claim_data['status']:
                status_class = "style='color: var(--danger-color)'"
            
            claim_type = "Independent" if claim_data.get('is_independent') else "Dependent"
            mpf = " (MPF)" if claim_data.get('is_means_plus_function') else ""
            
            html += f"""
                    <tr>
                        <td>{claim_data['application_number']}</td>
                        <td>{claim_data.get('issued_number') or '-'}</td>
"""
            
            if data['final_claims']:
                mapped = claim_data.get('mapped_final_claim') or '-'
                confidence = f"{claim_data.get('mapping_confidence', 0)*100:.0f}%" if claim_data.get('mapping_confidence') else '-'
                html += f"""                        <td>{mapped}</td>
                        <td>{confidence}</td>
"""
            
            html += f"""                        <td {status_class}>{claim_data['status']}</td>
                        <td>{claim_type}{mpf}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>C. Claim Element Lexicon</h2>
"""
        
        if data['lexicon']:
            for element, definitions in data['lexicon'].items():
                html += f"""
            <div class="lexicon-entry">
                <h4>{element}</h4>
"""
                # Layer 1: Synthesis narrative (if available)
                narrative = data.get('term_narratives', {}).get(element)
                if narrative:
                    html += f"""
                <div class="synthesis-box">
                    <strong>Synthesis:</strong> {narrative['summary']}
                </div>
"""
                    status = narrative.get('status', 'Unknown')
                    if status == 'Contradictory':
                        html += f"""
                <div class="consistency-warning">
                    <strong>⚠️ Inconsistency Detected:</strong> {narrative.get('contradiction_details', 'See raw evidence below.')}
                </div>
"""
                    elif status == 'Evolving':
                        html += f"""
                <div class="consistency-warning" style="border-left-color: var(--accent-color);">
                    <strong>ℹ️ Definition Evolved:</strong> {narrative.get('contradiction_details', 'The definition was refined over time.')}
                </div>
"""
                    elif status == 'Consistent':
                        html += """
                <div class="consistency-ok">
                    ✓ Consistent definition throughout prosecution
                </div>
"""
                
                # Layer 2: Raw data evidence
                html += """
                <div class="raw-evidence">
                    <p><strong>Raw Evidence:</strong></p>
"""
                for defn in definitions:
                    acquiesced = '<span class="acquiesced">✓ Acquiesced</span>' if defn.get('is_acquiesced') else ''
                    html += f"""
                    <p><strong>{defn.get('speaker', 'Unknown')}</strong> {acquiesced}</p>
                    <blockquote>{defn.get('definition', 'N/A')}</blockquote>
"""
                html += """
                </div>
            </div>"""
        else:
            html += '<p>No claim term definitions extracted.</p>'
        
        # Gap 1: Term Scope Boundaries
        term_boundaries = data.get('term_boundaries', {})
        if term_boundaries:
            html += """
            <h3>Term Scope Boundaries</h3>
            <p>The following boundaries define what specific claim terms include and exclude, based on prosecution statements:</p>
"""
            for term, bounds in term_boundaries.items():
                html += f'<div class="card"><strong>{term}</strong><br>'
                includes = [b for b in bounds if b['boundary_type'] == 'includes']
                excludes = [b for b in bounds if b['boundary_type'] == 'excludes']
                if includes:
                    html += '<em>Includes:</em><ul>'
                    for b in includes:
                        src = f' — <em>"{b["source_text"][:120]}"</em>' if b.get('source_text') else ''
                        html += f'<li>{b["example_text"]}{src}</li>'
                    html += '</ul>'
                if excludes:
                    html += '<em>Excludes:</em><ul>'
                    for b in excludes:
                        src = f' — <em>"{b["source_text"][:120]}"</em>' if b.get('source_text') else ''
                        html += f'<li>{b["example_text"]}{src}</li>'
                    html += '</ul>'
                html += '</div>'

        # Claim Evolution
        html += """
        </div>
        
        <div class="section">
            <h2>D. Claim Evolution</h2>
"""
        
        # Gap 4 fix: Use same prioritization as markdown
        all_html_claims = data['claims']
        html_issued = sorted(
            [c for c in all_html_claims if c.get('mapped_final_claim') is not None],
            key=lambda c: _int_sort_key(c.get('mapped_final_claim', 999))
        )
        html_other_indep = sorted(
            [c for c in all_html_claims
             if c.get('is_independent')
             and c.get('mapped_final_claim') is None
             and c.get('status') in ('Allowed', 'Pending', 'Rejected')],
            key=lambda c: _int_sort_key(c['application_number'])
        )
        html_claims_to_show = html_issued + html_other_indep
        html_seen = set()
        html_deduped = []
        for c in html_claims_to_show:
            if c['application_number'] not in html_seen:
                html_seen.add(c['application_number'])
                html_deduped.append(c)
        html_claims_to_show = html_deduped[:10]
        
        for claim in html_claims_to_show:
            header = f"Claim {claim['application_number']}"
            if claim.get('issued_number'):
                header += f" → Issued Claim {claim['issued_number']}"
            if claim.get('mapped_final_claim'):
                header += f" (Final: {claim['mapped_final_claim']})"
            
            html += f"""
            <h3>{header}</h3>
"""
            # Claim narrative (Layer 2 synthesis)
            narrative = data.get('claim_narratives', {}).get(claim['application_number'])
            if narrative:
                html += f"""
            <div class="narrative-box">
                <p>{narrative['evolution_summary']}</p>
"""
                if narrative.get('turning_point_event'):
                    html += f"""
                <p class="turning-point">🔑 Turning Point: {narrative['turning_point_event']}</p>
"""
                html += "            </div>\n"
            for v in claim.get('versions', []):
                html += f"""
            <p><strong>Version {v['version']}</strong> ({v.get('date', 'N/A')})</p>
"""
                if v.get('change_summary'):
                    html += f'<p><em>Change: {v["change_summary"]}</em></p>'
                
                text = v.get('text', 'N/A')
                html += f'<div class="claim-text">{text}</div>'
        
        # Gap 5: Vulnerability Cards
        vuln_cards = data.get('vulnerability_cards', {})
        if vuln_cards:
            html += """
        </div>
        
        <div class="section">
            <h2>E. Claim Vulnerability Assessment</h2>
            <p>Per-claim vulnerability cards synthesizing estoppel bars, categorical exclusions, and design-around paths.</p>
"""
            for claim_num in sorted(vuln_cards.keys(), key=_int_sort_key):
                card = vuln_cards[claim_num]
                overall = card.get('overall_vulnerability', 'Unknown')
                color = {'High': '#dc3545', 'Medium': '#ffc107', 'Low': '#28a745'}.get(overall, '#6c757d')
                html += f'<div class="card"><h3>Claim {claim_num} — <span style="color:{color}">{overall} Vulnerability</span></h3>'
                for section_key, section_label in [
                    ('must_practice', 'Must-Practice Elements'),
                    ('categorical_exclusions', 'Categorical Exclusions (Estoppel)'),
                    ('estoppel_bars', 'Estoppel Bars'),
                    ('indefiniteness_targets', 'Indefiniteness Targets'),
                    ('design_around_paths', 'Design-Around Paths'),
                ]:
                    items = card.get(section_key, [])
                    if items:
                        html += f'<strong>{section_label}:</strong><ul>'
                        for item in items:
                            html += f'<li>{item}</li>'
                        html += '</ul>'
                html += '</div>'

        # Gap 4: Prior Art References
        prior_art_refs = data.get('prior_art_refs', [])
        if prior_art_refs:
            html += """
            <h3>Consolidated Prior Art References</h3>
            <table><tr><th>Reference</th><th>Number</th><th>Basis</th><th>Claims</th><th>Status</th></tr>
"""
            for ref in prior_art_refs:
                status = '✅ Overcome' if ref.get('is_overcome') else '⚠️ Active'
                claims = ', '.join(str(c) for c in (ref.get('affected_claims') or []))
                html += f"<tr><td>{ref.get('canonical_name', '')}</td><td>{ref.get('patent_or_pub_number', '')}</td>"
                html += f"<td>{ref.get('applied_basis', '')}</td><td>{claims}</td><td>{status}</td></tr>"
            html += '</table>'

        # Prosecution Timeline (Gap 6 fix)
        html += """
        </div>
        
        <div class="section">
            <h2>F. Prosecution Timeline</h2>
            <div class="timeline">
"""
        
        # Sort by date with page-order fallback
        html_docs = [(i, doc) for i, doc in enumerate(data['documents'])]
        html_docs_sorted = sorted(
            html_docs,
            key=lambda x: (x[1].get('date') or '9999-99-99', x[0])
        )
        
        for _idx, doc in html_docs_sorted:
            priority_class = "high-priority" if doc.get('is_high_priority') else ""
            star = "⭐ " if doc.get('is_high_priority') else ""
            date = doc.get('date', 'Unknown')
            if date and len(date) >= 10:
                date = date[:10]
            elif not date:
                date = 'Unknown'
            pages = f" (pp. {doc['pages']})" if doc.get('pages') else ""
            html += f"""
                <div class="timeline-item {priority_class}">
                    <strong>{date}</strong>: {star}{doc.get('type', 'Unknown')}{pages}
                </div>
"""
        
        # Prosecution Milestones
        milestones = data.get('milestones', [])
        if milestones:
            html += '<h3>Prosecution Milestones</h3><ul>'
            for m in milestones:
                date_str = m.get('date', 'Unknown')
                if date_str and len(date_str) >= 10:
                    date_str = date_str[:10]
                m_type = m.get('type', 'Unknown')
                emoji = {'RCE': '🔄', 'Appeal': '⚖️', 'Continuation': '🔗', 'Petition': '📋'}.get(m_type, '📌')
                context = f" — {m['context']}" if m.get('context') else ""
                html += f'<li><strong>{date_str}</strong>: {emoji} {m_type}{context}</li>'
            html += '</ul>'

        html += """
            </div>
        </div>
        
        <footer>
            <p>Generated by PHAT - Patent Prosecution History Analysis Tool</p>
        </footer>
    </div>
</body>
</html>
"""
        
        return html
