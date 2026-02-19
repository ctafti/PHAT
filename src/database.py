"""
PHAT Database Models
SQLAlchemy ORM models for Patent Prosecution History Analysis
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime, Enum, ForeignKey, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import enum
import uuid

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class ClaimStatus(enum.Enum):
    PENDING = "Pending"
    OBJECTED_TO = "Objected To"
    REJECTED = "Rejected"
    CANCELLED = "Cancelled"
    ALLOWED = "Allowed"
    WITHDRAWN_NON_ELECTED = "Withdrawn - Non-Elected"


class Speaker(enum.Enum):
    APPLICANT = "Applicant"
    EXAMINER = "Examiner"
    PTAB = "PTAB"


class RelevanceCategory(enum.Enum):
    CLAIM_CONSTRUCTION = "Claim Construction"
    ESTOPPEL = "Estoppel"
    NON_INFRINGEMENT = "Non-Infringement"
    PRIOR_ART_DISTINCTION = "Prior Art Distinction"
    DEFINITION = "Definition/Interpretation"
    REASONS_FOR_ALLOWANCE = "Reasons for Allowance"
    INTERVIEW_CONCESSION = "Interview Concession"


class RejectionType(enum.Enum):
    ART_BASED = "Art_Based"
    FORMAL_OBJECTION = "Formal_Objection"
    STATUTORY_SUBJECT_MATTER = "Statutory_Subject_Matter"
    DOUBLE_PATENTING = "Double_Patenting"


class SupportStatus(enum.Enum):
    SUPPORTED = "Supported"
    UNSUPPORTED = "Unsupported"
    PARTIAL = "Partial"


class Patent(Base):
    """Master table for the patent being analyzed"""
    __tablename__ = 'patents'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_number = Column(String(50), nullable=True)
    application_number = Column(String(50), nullable=True)
    filing_date = Column(DateTime, nullable=True)
    issue_date = Column(DateTime, nullable=True)
    title = Column(Text, nullable=True)
    priority_date = Column(DateTime, nullable=True)
    provisional_number = Column(String(50), nullable=True)
    
    # Final claims from Google Patents (raw text)
    final_claims_text = Column(Text, nullable=True)
    # Final claims parsed as JSON (for quick access)
    final_claims_json = Column(JSON, nullable=True)
    # Flag indicating if final claims have been provided
    has_final_claims = Column(Boolean, default=False)
    
    # Relationships
    documents = relationship("Document", back_populates="patent", cascade="all, delete-orphan")
    claims = relationship("Claim", back_populates="patent", cascade="all, delete-orphan")
    terminal_disclaimers = relationship("TerminalDisclaimer", back_populates="patent", cascade="all, delete-orphan")
    final_claims = relationship("FinalClaim", back_populates="patent", cascade="all, delete-orphan")
    
    # Layer 2/3 Synthesis & Critique tables (additive — never modify raw data)
    validity_risks = relationship("ValidityRisk", back_populates="patent", cascade="all, delete-orphan")
    term_syntheses = relationship("TermSynthesis", back_populates="patent", cascade="all, delete-orphan")
    claim_narratives = relationship("ClaimNarrative", back_populates="patent", cascade="all, delete-orphan")
    patent_themes = relationship("PatentTheme", back_populates="patent", cascade="all, delete-orphan")
    
    # Gap Analysis v2 tables
    term_boundaries = relationship("TermBoundary", back_populates="patent", cascade="all, delete-orphan")
    prior_art_references = relationship("PriorArtReference", back_populates="patent", cascade="all, delete-orphan")
    vulnerability_cards = relationship("ClaimVulnerabilityCard", back_populates="patent", cascade="all, delete-orphan")
    milestones = relationship("ProsecutionMilestone", back_populates="patent", cascade="all, delete-orphan")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FinalClaim(Base):
    """Stores final issued claims from Google Patents as ground truth"""
    __tablename__ = 'final_claims'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    
    # Claim number in the issued patent
    claim_number = Column(Integer, nullable=False)
    # Full claim text
    claim_text = Column(Text, nullable=False)
    # Normalized text for comparison
    normalized_text = Column(Text, nullable=True)
    
    # Claim structure
    is_independent = Column(Boolean, default=True)
    depends_on = Column(Integer, nullable=True)  # Claim number this depends on
    
    # Gap 12: Claim type derived from preamble
    claim_type = Column(String(50), nullable=True)  # "Method", "System", "Computer-Readable Medium", "Apparatus", "Composition", "Unknown"
    
    # Mapping to application claims (if determined)
    mapped_app_claim_id = Column(String(36), ForeignKey('claims.id'), nullable=True)
    mapping_confidence = Column(Float, nullable=True)  # 0.0 to 1.0
    
    # Relationships
    patent = relationship("Patent", back_populates="final_claims")
    mapped_app_claim = relationship("Claim", foreign_keys=[mapped_app_claim_id])
    
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    """Tracks each document in the file history"""
    __tablename__ = 'documents'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    document_type = Column(String(100), nullable=False)  # Amendment, Office Action, etc.
    document_date = Column(DateTime, nullable=True)
    mail_date = Column(DateTime, nullable=True)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    filename = Column(String(255), nullable=True)
    is_high_priority = Column(Boolean, default=False)
    is_processed = Column(Boolean, default=False)
    raw_text = Column(Text, nullable=True)
    
    # Relationships
    patent = relationship("Patent", back_populates="documents")
    claim_versions = relationship("ClaimVersion", back_populates="document", cascade="all, delete-orphan")
    statements = relationship("ProsecutionStatement", back_populates="document", cascade="all, delete-orphan")
    rejections = relationship("RejectionHistory", back_populates="document", cascade="all, delete-orphan")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class Claim(Base):
    """Master claim identity table - tracks claim genealogy"""
    __tablename__ = 'claims'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    application_claim_number = Column(Integer, nullable=False)
    final_issued_number = Column(Integer, nullable=True)
    family_tree_id = Column(String(36), nullable=True)  # Links dependent to independent parent
    parent_claim_id = Column(String(36), ForeignKey('claims.id'), nullable=True)
    is_independent = Column(Boolean, default=True)
    is_means_plus_function = Column(Boolean, default=False)
    means_plus_function_elements = Column(JSON, nullable=True)  # List of elements invoking 112(f)
    current_status = Column(String(50), default=ClaimStatus.PENDING.value)
    
    # Restriction requirement tracking
    elected_group = Column(String(50), nullable=True)
    is_elected = Column(Boolean, default=True)
    
    # Final claim mapping
    mapped_final_claim_number = Column(Integer, nullable=True)
    mapping_confidence = Column(Float, nullable=True)
    
    # Relationships
    patent = relationship("Patent", back_populates="claims")
    versions = relationship("ClaimVersion", back_populates="claim", cascade="all, delete-orphan")
    children = relationship("Claim", backref="parent", remote_side=[id])
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ClaimVersion(Base):
    """Tracks text history of each claim"""
    __tablename__ = 'claim_versions'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    claim_id = Column(String(36), ForeignKey('claims.id'), nullable=False)
    document_id = Column(String(36), ForeignKey('documents.id'), nullable=False)
    version_number = Column(Integer, default=1)
    claim_text = Column(Text, nullable=False)
    normalized_text = Column(Text, nullable=True)  # Stripped of whitespace/punctuation for comparison
    change_summary = Column(Text, nullable=True)
    is_normalized_change = Column(Boolean, default=True)  # False if only whitespace/typo change
    date_of_change = Column(DateTime, nullable=True)
    added_limitations = Column(JSON, nullable=True)
    
    # Gap 3: Track who authored the amendment for Festo analysis
    amendment_source = Column(String(20), nullable=True)  # "applicant" | "examiner" | "examiner_interview"
    
    # Relationships
    claim = relationship("Claim", back_populates="versions")
    document = relationship("Document", back_populates="claim_versions")
    addressed_rejections = relationship("RejectionHistory", back_populates="addressed_by_version")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class ProsecutionStatement(Base):
    """Captures definitions, arguments, and legal mandates"""
    __tablename__ = 'prosecution_statements'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey('documents.id'), nullable=False)
    speaker = Column(String(20), nullable=False)  # Applicant, Examiner, PTAB
    extracted_text = Column(Text, nullable=False)
    relevance_category = Column(String(50), nullable=True)
    is_acquiesced = Column(Boolean, default=False)
    traversal_present = Column(Boolean, default=False)
    claim_element_defined = Column(String(255), nullable=True)
    affected_claims = Column(JSON, nullable=True)  # List of claim numbers
    cited_prior_art = Column(JSON, nullable=True)  # List of prior art references
    context_summary = Column(Text, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="statements")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class RejectionHistory(Base):
    """Tracks rejections and their resolutions"""
    __tablename__ = 'rejection_history'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey('documents.id'), nullable=False)
    claim_number = Column(Integer, nullable=False)  # Claim number at time of rejection
    rejection_type = Column(String(50), nullable=False)
    statutory_basis = Column(String(50), nullable=True)  # 102, 103, 112, etc.
    cited_prior_art = Column(JSON, nullable=True)
    rejected_claim_elements = Column(JSON, nullable=True)
    rejection_rationale = Column(Text, nullable=True)
    is_final = Column(Boolean, default=False)
    addressed_by_version_id = Column(String(36), ForeignKey('claim_versions.id'), nullable=True)
    is_overcome = Column(Boolean, default=False)
    
    # Relationships
    document = relationship("Document", back_populates="rejections")
    addressed_by_version = relationship("ClaimVersion", back_populates="addressed_rejections")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class PriorityBreak(Base):
    """Tracks potential priority date issues"""
    __tablename__ = 'priority_breaks'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    document_id = Column(String(36), ForeignKey('documents.id'), nullable=False)
    feature_element = Column(String(255), nullable=False)
    added_date = Column(DateTime, nullable=True)
    unsupported_date = Column(DateTime, nullable=True)
    support_status = Column(String(20), default=SupportStatus.UNSUPPORTED.value)
    analysis_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class TerminalDisclaimer(Base):
    """Tracks terminal disclaimers"""
    __tablename__ = 'terminal_disclaimers'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    document_id = Column(String(36), ForeignKey('documents.id'), nullable=True)
    disclaimed_patent = Column(String(50), nullable=True)
    disclaimer_date = Column(DateTime, nullable=True)
    reason = Column(Text, nullable=True)
    
    # Relationships
    patent = relationship("Patent", back_populates="terminal_disclaimers")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class RestrictionRequirement(Base):
    """Tracks restriction requirements and elections"""
    __tablename__ = 'restriction_requirements'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    document_id = Column(String(36), ForeignKey('documents.id'), nullable=True)
    restriction_date = Column(DateTime, nullable=True)
    groups = Column(JSON, nullable=True)  # Dict of group names to claim numbers
    elected_group = Column(String(50), nullable=True)
    elected_claims = Column(JSON, nullable=True)
    non_elected_claims = Column(JSON, nullable=True)
    traversed = Column(Boolean, default=False)
    traversal_arguments = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class ValidityRisk(Base):
    """
    The Shadow Examiner: stores forward-looking validity critiques of allowed claims.
    
    This is a Layer 3 (Critique) table — purely derived intelligence that does NOT
    modify or replace any raw extraction data.
    """
    __tablename__ = 'validity_risks'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    
    # Analysis
    risk_type = Column(String(50), nullable=False)     # e.g., "Mixed Statutory Class", "112(b) Indefiniteness"
    severity = Column(String(20), default="Medium")    # High, Medium, Low
    description = Column(Text, nullable=False)         # The AI's critique
    reasoning = Column(Text, nullable=True)            # Why this is a risk
    
    # Linkage to Raw Data
    affected_claims = Column(JSON, nullable=True)      # List of claim numbers [1, 5]
    
    # Relationships
    patent = relationship("Patent", back_populates="validity_risks")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class TermSynthesis(Base):
    """
    Synthesized definition narratives and flip-flop consistency checks.
    
    This is a Layer 2 (Synthesis) table — derives a cohesive narrative from
    multiple ProsecutionStatement rows grouped by claim_element_defined.
    """
    __tablename__ = 'term_synthesis'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    
    term = Column(String(255), nullable=False)
    
    # The Narrative
    narrative_summary = Column(Text, nullable=False)   # Cohesive paragraph
    
    # The Consistency Check
    consistency_status = Column(String(50))            # "Consistent", "Contradictory", "Evolving"
    contradiction_details = Column(Text, nullable=True) # Explanation of the flip-flop
    
    # Linkage to Raw Data (crucial for user trust / chain of custody)
    source_statement_ids = Column(JSON, nullable=True) # List of ProsecutionStatement IDs
    
    # Relationships
    patent = relationship("Patent", back_populates="term_syntheses")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class ClaimNarrative(Base):
    """
    The "biography" of a specific independent claim — its story arc through prosecution.
    
    This is a Layer 2 (Synthesis) table.
    """
    __tablename__ = 'claim_narratives'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    
    claim_number = Column(Integer, nullable=False)
    
    # The Story
    evolution_summary = Column(Text, nullable=False)   # "Claim 1 was allowed after adding..."
    turning_point_event = Column(Text, nullable=True)  # "Amendment filed 2018-05-12"
    
    # Relationships
    patent = relationship("Patent", back_populates="claim_narratives")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class PatentTheme(Base):
    """
    Thematic synthesis of prosecution arguments and amendments.
    
    Groups multiple arguments/amendments under overarching "Themes of Patentability"
    (e.g., "Integrity of Original Images", "Two-Step Location Refinement").
    
    This is a Layer 2 (Synthesis) table — Improvement A.
    """
    __tablename__ = 'patent_themes'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    
    # Theme identity
    title = Column(String(255), nullable=False)              # Short descriptive title
    summary = Column(Text, nullable=False)                   # 2-4 sentence explanation
    
    # Supporting evidence
    key_arguments = Column(JSON, nullable=True)              # List of argument summaries
    key_amendments = Column(JSON, nullable=True)             # List of amendment summaries
    prior_art_distinguished = Column(JSON, nullable=True)    # List of reference names
    affected_claims = Column(JSON, nullable=True)            # List of claim numbers
    
    # Litigation significance
    estoppel_significance = Column(Text, nullable=True)      # What scope is surrendered
    metaphors_or_analogies = Column(JSON, nullable=True)     # Conceptual characterizations
    
    # Relationships
    patent = relationship("Patent", back_populates="patent_themes")
    
    created_at = Column(DateTime, default=datetime.utcnow)



# =============================================================================
# GAP ANALYSIS V2: New Models
# =============================================================================

class TermBoundary(Base):
    """
    Gap 1: Stores specific enumerated examples of what falls inside/outside
    a prosecution-defined term's scope.
    
    For example, if applicant argues "without modification" excludes filtering,
    truncating, and resampling, each of those is a separate boundary entry.
    """
    __tablename__ = 'term_boundaries'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    term_synthesis_id = Column(String(36), ForeignKey('term_synthesis.id'), nullable=True)
    
    term = Column(String(255), nullable=False)         # The term being bounded
    boundary_type = Column(String(20), nullable=False)  # "includes" | "excludes"
    example_text = Column(Text, nullable=False)          # The specific enumerated item
    source_text = Column(Text, nullable=True)            # The original prosecution statement
    source_statement_id = Column(String(36), ForeignKey('prosecution_statements.id'), nullable=True)
    affected_claims = Column(JSON, nullable=True)
    
    # Relationships
    patent = relationship("Patent", back_populates="term_boundaries")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class PriorArtReference(Base):
    """
    Gap 4: Normalized, deduplicated, queryable table of prior art references.
    Consolidates scattered JSON fields into a structured format needed for
    Phase 2 prior art retrieval and analysis.
    """
    __tablename__ = 'prior_art_references'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    
    # Identity
    canonical_name = Column(String(255), nullable=False)    # Normalized: "Kato"
    patent_or_pub_number = Column(String(100), nullable=True)  # "US 5,982,378"
    inventor_names = Column(JSON, nullable=True)
    
    # How it was used
    applied_basis = Column(JSON, nullable=True)              # ["102", "103_primary"]
    affected_claims = Column(JSON, nullable=True)            # Union of all claims
    
    # Prosecution outcome
    key_teachings = Column(JSON, nullable=True)              # What examiner says it teaches
    key_deficiencies = Column(JSON, nullable=True)           # What applicant says it lacks
    is_overcome = Column(Boolean, default=False)
    
    # Metadata for retrieval
    filing_date = Column(DateTime, nullable=True)
    title = Column(Text, nullable=True)
    
    # Relationships
    patent = relationship("Patent", back_populates="prior_art_references")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class ClaimVulnerabilityCard(Base):
    """
    Gap 5: Per-claim unified vulnerability assessment that a litigator
    can use directly. Synthesizes estoppel events, validity risks,
    scope constraints, and prosecution positions into a single view.
    """
    __tablename__ = 'claim_vulnerability_cards'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    claim_number = Column(Integer, nullable=False)
    
    # Synthesized from multiple sources
    must_practice = Column(JSON, nullable=True)            # Affirmative requirements
    categorical_exclusions = Column(JSON, nullable=True)   # Things categorically excluded
    estoppel_bars = Column(JSON, nullable=True)            # Equivalents foreclosed
    indefiniteness_targets = Column(JSON, nullable=True)   # Vulnerable language
    design_around_paths = Column(JSON, nullable=True)      # Suggested escape routes
    overall_vulnerability = Column(String(20), nullable=True)  # High/Medium/Low
    
    # Relationships
    patent = relationship("Patent", back_populates="vulnerability_cards")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class ProsecutionMilestone(Base):
    """
    Gap 9: Tracks RCE filings, appeals, and continuation events as
    procedural milestones in the prosecution timeline.
    """
    __tablename__ = 'prosecution_milestones'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    
    milestone_type = Column(String(50), nullable=False)  # "RCE", "Appeal", "Continuation", "Petition"
    date = Column(DateTime, nullable=True)
    context = Column(Text, nullable=True)  # Brief note
    document_id = Column(String(36), ForeignKey('documents.id'), nullable=True)
    
    # Relationships
    patent = relationship("Patent", back_populates="milestones")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class AnalysisRun(Base):
    """Tracks analysis runs for audit purposes"""
    __tablename__ = 'analysis_runs'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patent_id = Column(String(36), ForeignKey('patents.id'), nullable=False)
    ai_provider = Column(String(50), nullable=False)
    ai_model = Column(String(100), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String(20), default="running")
    documents_processed = Column(Integer, default=0)
    errors = Column(JSON, nullable=True)
    
    # Track if final claims were used
    used_final_claims = Column(Boolean, default=False)
    
    # Track if post-processing was run
    post_processing_completed = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self, db_path: str):
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        return self.Session()
    
    def create_tables(self):
        Base.metadata.create_all(self.engine)
    
    def drop_tables(self):
        Base.metadata.drop_all(self.engine)
    
    def store_final_claims(self, patent_id: str, claims_text: str, parsed_claims: list) -> None:
        """Store final claims for a patent"""
        session = self.Session()
        try:
            patent = session.query(Patent).get(patent_id)
            if not patent:
                raise ValueError(f"Patent not found: {patent_id}")
            
            # Store raw text and JSON
            patent.final_claims_text = claims_text
            patent.final_claims_json = [
                {
                    'number': c.number,
                    'text': c.text,
                    'is_independent': c.is_independent,
                    'depends_on': c.depends_on
                }
                for c in parsed_claims
            ]
            patent.has_final_claims = True
            
            # Clear any existing final claims
            session.query(FinalClaim).filter(FinalClaim.patent_id == patent_id).delete()
            
            # Store individual claims
            for claim in parsed_claims:
                fc = FinalClaim(
                    patent_id=patent_id,
                    claim_number=claim.number,
                    claim_text=claim.text,
                    is_independent=claim.is_independent,
                    depends_on=claim.depends_on
                )
                session.add(fc)
            
            session.commit()
            
        finally:
            session.close()
    
    def get_patent_status_by_filename(self, filename: str) -> tuple:
        """
        Check if a file has already been processed or started.
        
        Looks up the Patent ID associated with a specific PDF filename
        by checking Document records. This enables filename-based resume
        without relying on OCR/regex extraction of patent info.
        
        Args:
            filename: The PDF filename (e.g., "US Patent No. 8,026,929 - File History.pdf")
            
        Returns:
            (patent_id, status_string) or (None, None) if not found.
            Status comes from the most recent AnalysisRun for that patent.
        """
        session = self.Session()
        try:
            # Find a document record matching this filename
            doc = session.query(Document).filter(
                Document.filename == filename
            ).first()
            
            if not doc:
                return None, None
            
            # Check the latest AnalysisRun for this patent
            run = session.query(AnalysisRun).filter(
                AnalysisRun.patent_id == doc.patent_id
            ).order_by(AnalysisRun.start_time.desc()).first()
            
            status = run.status if run else "unknown"
            
            return doc.patent_id, status
            
        finally:
            session.close()
    
    def get_final_claims(self, patent_id: str) -> list:
        """Get final claims for a patent"""
        session = self.Session()
        try:
            claims = session.query(FinalClaim).filter(
                FinalClaim.patent_id == patent_id
            ).order_by(FinalClaim.claim_number).all()
            
            return [
                {
                    'number': c.claim_number,
                    'text': c.claim_text,
                    'is_independent': c.is_independent,
                    'depends_on': c.depends_on,
                    'mapped_app_claim_id': c.mapped_app_claim_id,
                    'mapping_confidence': c.mapping_confidence
                }
                for c in claims
            ]
        finally:
            session.close()


if __name__ == "__main__":
    # Test database creation
    db = DatabaseManager("test_phat.db")
    session = db.get_session()
    
    # Create test patent
    patent = Patent(
        patent_number="US10,000,001",
        application_number="16/123,456",
        title="Test Patent"
    )
    session.add(patent)
    session.commit()
    
    print(f"Created patent with ID: {patent.id}")
    session.close()
