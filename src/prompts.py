"""
PHAT Analysis Prompts
Centralized prompt templates for AI analysis tasks
"""

SYSTEM_PROMPT = """You are a patent prosecution analyst expert. Your task is to analyze patent file history documents with precision and accuracy. You must:

1. Extract information accurately from the provided document text
2. Identify claim amendments, definitions, and legal arguments
3. Track claim genealogy and renumbering
4. Identify examiner rejections and applicant responses
5. Flag important legal events (terminal disclaimers, means-plus-function claims, etc.)

Always respond with valid JSON as specified in the task. Be precise and thorough."""


# Task 1: Document Triage
TRIAGE_PROMPT = """Analyze this document excerpt and determine if it contains legally significant content for patent prosecution analysis.

Document Text:
{document_text}

Answer with a JSON object:
{{
    "is_relevant": true/false,
    "document_type": "string - type of document (Office Action, Amendment, etc.)",
    "contains_claims": true/false,
    "contains_rejections": true/false,
    "contains_arguments": true/false,
    "contains_definitions": true/false,
    "contains_restriction": true/false,
    "contains_allowance": true/false,
    "contains_terminal_disclaimer": true/false,
    "priority_level": "high/medium/low",
    "summary": "brief 1-2 sentence summary of document"
}}"""


# Task 2: Initial Claims Extraction
CLAIMS_EXTRACTION_PROMPT = """Extract all claims from this patent document text. For each claim, identify:
- The claim number
- Whether it is independent or dependent
- If dependent, which claim it depends on
- The full claim text

Document Text:
{document_text}

Respond with JSON:
{{
    "claims": [
        {{
            "number": 1,
            "is_independent": true,
            "depends_on": null,
            "text": "full claim text here"
        }},
        {{
            "number": 2,
            "is_independent": false,
            "depends_on": 1,
            "text": "full claim text here"
        }}
    ]
}}"""


# Task 3: Combined Amendment + Argument Analysis (Estoppel Linker)
# Extracts amendments AND the arguments distinguishing them in one pass.
# This replaces the need to run Argument Extraction separately for Response docs,
# saving tokens and forcing the AI to explicitly link amendments to arguments.
AMENDMENT_ANALYSIS_PROMPT = """Analyze this amendment/response document.

Previous Claims State:
{previous_claims}

Document Text:
{document_text}

Prior Art from Rejection:
{prior_art_list}

TASK:
1. Identify structural changes to claims (Amended/New/Cancelled).
2. For every amendment, extract the specific "Added Limitations" (new text).
3. CRITICAL: For each added limitation, extract the SPECIFIC ARGUMENT used to distinguish it from the Prior Art.
   - Look for phrases like "Applicant submits that [Reference] fails to teach..."
   - Identify if this creates "Prosecution History Estoppel" (narrowing claim scope to overcome art).
4. Extract any GENERAL ARGUMENTS (traversals, procedural objections, motivations to combine) that are not tied to a specific claim text change.
5. Extract THEMATIC / CONCEPTUAL DISTINCTIONS: broad positions about what the invention IS and IS NOT.
   These are critically important for litigation. Look for:
   - "The invention is NOT directed to [X]. Rather, it [Y]." 
   - "Using images as [X] is fundamentally different from [Y]."
   - "[Reference] violates the integrity of..." / "does not respect..."
   - Characterizations of the invention's purpose, function, or nature
     (e.g., "the 3D model serves as a catalog" — purpose/function characterizations).
   - "Unlike [Reference], the claimed invention..." / "the goal is exactly the opposite"
   - Directionality arguments (e.g., "snapping of A to B, not B to A", "indicators of where A 
     is located within B, not where B is located within A")
   - Categorical exclusions (e.g., "the 2D images cannot be used to create the 3D model",
     "the 2D image cannot be used as a surface texture")
   - **Conceptual Analogies and Metaphors** (e.g., "the model acts as a catalog", 
     "images hover in ether", "a digital filing cabinet"). Look for keywords: "metaphor", 
     "essence", "akin to", "serves as", "acts like", "functions as", "analogous to".
     These are HIGH VALUE — tag them as statement_type "conceptual_analogy".
   Include these as general_arguments with statement_type "thematic_distinction"
   (or "conceptual_analogy" for metaphors and analogies).
   
   For DIRECTIONALITY arguments specifically, also set "directionality_constraint": true and 
   include a clear statement of the required direction (e.g., "3D snaps to 2D pose, not 2D to 3D").
   
   For PURPOSE/FUNCTION characterizations, set statement_type to "purpose_characterization".

6. PRACTICAL IMPLICATIONS: For each added limitation, identify 1-2 practical constraints
   it places on any product that would practice this claim. Write in plain engineering
   language, not legal jargon. Consider:
   - Hardware/software requirements (e.g., "Requires a GPS module", "Must include a 2D map display")
   - User interaction requirements (e.g., "User must manually click/select; cannot be automated")
   - Data format constraints (e.g., "Images must be stored unaltered — no cropping, warping, or masking")
   - Architectural constraints (e.g., "3D model must be pre-existing; cannot be generated from the 2D images")
   - Potential indefiniteness issues (e.g., "Claim recites user action in apparatus claim — mixed method-apparatus")

7. HOUSEKEEPING AMENDMENTS: Mark housekeeping-only amendments clearly. If a claim change consists ONLY of:
   - Correcting a typo or grammatical error
   - Adding "Non-transitory" to a preamble for 101 compliance
   - Renumbering due to cancellation of other claims
   - Simple word substitution without scope change (e.g., "map" to "match")
   Then set "is_housekeeping": true on the amendment entry. These will be deprioritized in
   the report. Only amendments that ADD NEW LIMITATIONS or CHANGE CLAIM SCOPE should have
   is_housekeeping: false (the default).

*** MANDATORY RULE FOR "current_text" ***
For EVERY claim with change_type "amended" or "new", you MUST include the COMPLETE, FULL claim 
text in the "current_text" field — the entire claim as it reads AFTER the amendment, word for word.
DO NOT summarize it. DO NOT omit it. DO NOT leave it null or empty.
If the amendment document shows the claim with markings (underline for additions, brackets for 
deletions), reconstruct the clean final text by applying those changes.
If you cannot find the full amended text, copy the previous claim text and apply the changes 
described in the change_summary to produce the best reconstruction you can.
Omitting current_text makes the entire analysis useless.
***

*** TOKEN MANAGEMENT ***
If you are running out of token space, prioritize finishing the current sentence and closing 
the current JSON structure cleanly over starting a new point. A complete analysis of fewer 
items is far more useful than a truncated analysis of many items.
***

Respond with JSON:
{{
    "_reasoning": "Identify which claims have been amended, cancelled, or added. For each amendment, identify the specific text that was added and the prior art it responds to before filling the fields below.",
    "document_date": "YYYY-MM-DD (the filing or mailing date found in the document header)",
    "amended_claims": [
        {{
            "claim_number": 1,
            "change_type": "amended/cancelled/new",
            "previous_text": "text before amendment or null if new",
            "current_text": "MANDATORY — complete full claim text after amendment, or null ONLY if cancelled",
            "change_summary": "brief description of what changed",
            "added_limitations": [
                {{
                    "text": "the exact phrase added (e.g., 'ground truth 3D model')",
                    "likely_response_to_art": "Name of prior art reference (e.g., 'Debevec') or 'None'",
                    "applicant_argument_summary": "Specific argument: Smith lacks this feature because...",
                    "estoppel_risk": "High/Medium/Low",
                    "practical_implications": [
                        "Plain-English constraint on product design (e.g., 'System must include a 2D geographic map UI')",
                        "Second constraint if applicable (e.g., 'Requires manual user interaction; cannot be fully automated')"
                    ],
                    "is_housekeeping": false
                }}
            ]
        }}
    ],
    "general_arguments": [
        {{
            "speaker": "Applicant",
            "text": "Quote of general traversal argument not tied to specific claim text change",
            "statement_type": "traversal/distinction/definition/scope_limitation/thematic_distinction/purpose_characterization",
            "claim_element_defined": "the specific claim term being defined or discussed (e.g., '3D model', 'without alteration', 'visual indicator'), or null if no specific term. Do NOT use generic labels like 'claims generally' or 'the invention'.",
            "context": "Argues 103 combination is improper due to lack of motivation",
            "cited_prior_art": "reference name or null",
            "affected_claims": [1, 2, 3],
            "estoppel_risk": "high/medium/low",
            "traversal_present": true,
            "directionality_constraint": false
        }}
    ],
    "renumbering": [
        {{
            "old_number": 5,
            "new_number": 3,
            "reason": "claims 2-4 cancelled"
        }}
    ]
}}"""


# Targeted retry prompt: When the main amendment analysis omits current_text,
# we send a focused follow-up asking ONLY for the full claim text.
# This is cheaper than re-running the full analysis.
AMENDMENT_TEXT_RETRY_PROMPT = """The previous analysis of an amendment document identified that the following claims were amended, but did not provide the full claim text after amendment.

Document Text (the same amendment document):
{document_text}

Previous claim text (BEFORE amendment):
{previous_claims_text}

Claims that need their FULL AMENDED TEXT extracted:
{missing_claims}

For each claim listed above, find the claim AS IT READS AFTER THE AMENDMENT in the document.
Patent amendment documents typically show amended claims with markings:
- Text in [[double brackets]] or [brackets] = DELETED
- Text that is underlined or in bold = ADDED
- Sometimes shown as clean rewritten claims

Reconstruct the CLEAN, FINAL text of each claim after applying the amendment.

Respond with JSON:
{{
    "claims": [
        {{
            "claim_number": 1,
            "current_text": "The complete claim text after amendment, with all additions applied and all deletions removed."
        }}
    ]
}}"""


# Task 4: Rejection Analysis
REJECTION_ANALYSIS_PROMPT = """Analyze this Office Action to extract all rejections and objections.

Document Text:
{document_text}

For each rejection/objection, identify:

CRITICAL: Distinguish between "Cited Art" (listed on Form 892 or Notice of References Cited but NOT used in any rejection) and "Applied Art" (explicitly used by the Examiner to reject claims under 102/103). Only include APPLIED ART in the `prior_art` array.

For 103 rejections specifically:
- Identify which reference is the "primary" reference and which is "secondary."
- Extract the examiner's specific "motivation to combine" the references.

1. The statutory basis (35 USC 102, 103, 112, etc.)
2. The claims affected
3. The prior art cited (if applicable)
4. The specific elements/limitations at issue
5. Whether it's a rejection (substantive) or objection (formal)
6. Whether it's final or non-final

Respond with JSON:
{{
    "is_final_action": true/false,
    "rejections": [
        {{
            "rejection_id": "unique identifier",
            "type": "rejection/objection",
            "statutory_basis": "102/103/112(a)/112(b)/double_patenting/etc",
            "affected_claims": [1, 2, 3],
            "prior_art": [
                {{
                    "reference": "Smith et al.",
                    "patent_or_pub_number": "US 9,000,000",
                    "relevance": "teaches element X"
                }}
            ],
            "rejected_elements": ["element A", "limitation B"],
            "examiner_rationale": "brief summary of examiner's reasoning",
            "is_103_combination": true/false,
            "motivation_to_combine": "examiner's stated rationale for combining references (103 only)"
        }}
    ],
    "objections": [
        {{
            "type": "claim_objection/specification_objection/drawing_objection",
            "affected_claims": [5],
            "reason": "antecedent basis issue"
        }}
    ]
}}"""


# Task 4b: Consolidated Office Action Analysis (replaces 3 separate calls for OA docs)
OFFICE_ACTION_MASTER_PROMPT = """Analyze this Office Action comprehensively.

Document Text:
{document_text}

TASKS:
1. Extract all Rejections and Objections (statutory basis, applied art, rationale).
2. Extract any Examiner definitions or claim interpretations stated in the Office Action.
3. Check for 35 USC 112(f) (Means-Plus-Function) issues raised by the Examiner.
4. Extract any claims indicated as allowable.

CRITICAL INSTRUCTIONS:
- Distinguish between "Cited Art" (listed on Form 892 but not applied) and "Applied Art" (actually used in a 102/103 rejection). Only extract APPLIED ART in the rejections array.
- For 103 rejections, identify the specific motivation to combine stated by the examiner.

Respond with JSON:
{{
    "_reasoning": "Briefly identify the type of Office Action (Final/Non-Final), the statutory bases used, and the key claims at issue before filling the fields below.",
    "document_date": "YYYY-MM-DD (the mailing date found in the document header)",
    "is_final_action": true/false,
    "rejections": [
        {{
            "rejection_id": "unique identifier",
            "type": "rejection/objection",
            "statutory_basis": "102/103/112(a)/112(b)/double_patenting/etc",
            "affected_claims": [1, 2, 3],
            "prior_art": [
                {{
                    "reference": "Smith et al.",
                    "patent_or_pub_number": "US 9,000,000",
                    "relevance": "teaches element X"
                }}
            ],
            "rejected_elements": ["element A", "limitation B"],
            "examiner_rationale": "brief summary of examiner's reasoning",
            "is_103_combination": true/false,
            "motivation_to_combine": "examiner's stated rationale for combining references (103 only)"
        }}
    ],
    "objections": [
        {{
            "type": "claim_objection/specification_objection/drawing_objection",
            "affected_claims": [5],
            "reason": "antecedent basis issue"
        }}
    ],
    "examiner_definitions": [
        {{
            "term": "network interface",
            "interpretation": "examiner interprets broadly to include...",
            "affected_claims": [1, 5]
        }}
    ],
    "means_plus_function_issues": [
        {{
            "claim_number": 1,
            "element_text": "means for processing...",
            "examiner_invoked_112f": true/false,
            "corresponding_structure": "processor configured to...",
            "specification_support": "paragraph numbers or description"
        }}
    ],
    "allowable_claims": [7, 8]
}}"""


# Task 5: Argument and Definition Extraction (with Negative Limitation Scanner)
ARGUMENT_EXTRACTION_PROMPT = """Extract legally significant arguments and definitions from this prosecution document.

Focus specifically on these HIGH-VALUE patterns:

1. **Negative Scope / Disavowal**: Phrases where Applicant defines what the invention is NOT.
   - "unlike Reference X...", "does not include...", "excludes...", "is distinct from..."
   - These create binding scope limitations in litigation.

For each potential disavowal, assess your confidence:
- "high": Uses explicit exclusionary language ("excludes", "does not cover", "invention is limited to", "never includes").
- "medium": Implies exclusion through distinction ("unlike Reference X, which uses...").
- "low": General distinction that does not clearly surrender scope.
Include `disavowal_confidence` and `disavowal_reasoning` for any statement where `negative_limitation` is true.

2. **Definitions / Interpretations**: Where Applicant defines claim terms.
   - "the term 'processor' is defined herein to mean..."
   - "as used in the claims, 'module' refers to..."
   - "Applicant uses the term 'model' to mean the database itself"
   - "without alteration" means no cropping, masking, or warping
   
   IMPORTANT: The `claim_element` field should be the SPECIFIC TECHNICAL TERM being defined
   (e.g., "3D model", "without alteration", "visual indicator", "ground truth", "pose data").
   Do NOT use generic labels like "the invention", "claims generally", "amended claims",
   "invention (general)", or "Not specified" — these are useless for claim construction.

3. **Prior Art Distinctions**: Arguments distinguishing the invention from cited references.
   - "Applicant submits that Smith fails to teach..."
   - "The combination of Smith and Jones does not render obvious..."

4. **Traversals**: Arguments objecting to examiner positions.
   - "Applicant respectfully traverses..."

5. **Concessions / Scope Limitations**: Statements that limit what the claims cover.
   - "Applicant does not claim...", "The invention is limited to..."

6. **Metaphors, Analogies, or Conceptual Characterizations**: Used by the Applicant to 
   describe the invention's nature or to create vivid conceptual distinctions from prior art.
   - "acts as a catalog", "hovers like a ghost", "digital ether", "filing cabinet"
   - "the images are never integrated into the model — they hover alongside it"
   - "serves as a visual index, not a rendered scene"
   - Look for keywords: "metaphor", "essence", "akin to", "serves as", "acts like", 
     "functions as", "analogous to", "spirit of", "conceptually", "in nature"
   - These are critically important for claim construction and estoppel because they reveal
     what the applicant believes the invention IS and IS NOT at a conceptual level.
   - Label these as statement_type: "conceptual_analogy"
   - Set `claim_element` to the specific term being characterized (e.g., "3D model", "images")
   - These should be given HIGH PRIORITY — they often capture the "spirit" of the invention
     better than dry technical definitions and are the most memorable framing for attorneys.

Do NOT extract:
- Generic legal boilerplate
- Status corrections or typos
- IDS discussions
- Fee-related statements

Document Text:
{document_text}

Respond with JSON:
{{
    "statements": [
        {{
            "speaker": "Applicant/Examiner/PTAB",
            "statement_type": "definition/distinction/disavowal/scope_limitation/concession/traversal/conceptual_analogy",
            "extracted_text": "exact quote from document",
            "claim_element": "the specific term being defined or discussed",
            "negative_limitation": true/false,
            "disavowal_confidence": "high/medium/low (only when negative_limitation is true)",
            "disavowal_reasoning": "explanation of why this is/isn't a clear disavowal (only when negative_limitation is true)",
            "affected_claims": [1, 2, 5],
            "cited_prior_art": "reference being distinguished if applicable",
            "context": "brief context of why this statement was made",
            "estoppel_risk": "high/medium/low",
            "traversal_present": true/false
        }}
    ]
}}"""


# Task 6: Restriction Requirement Analysis  
RESTRICTION_ANALYSIS_PROMPT = """Analyze this Restriction Requirement document.

Document Text:
{document_text}

Extract:
1. The groups identified by the examiner
2. Which claims belong to each group
3. The elected group (if election has been made)
4. Whether applicant traversed the restriction
5. Any linking claims identified

Respond with JSON:
{{
    "groups": [
        {{
            "group_name": "Group I",
            "invention_description": "brief description",
            "claims": [1, 2, 3]
        }}
    ],
    "elected_group": "Group I or null if not elected",
    "elected_claims": [1, 2, 3],
    "non_elected_claims": [4, 5, 6],
    "traversed": true/false,
    "traversal_arguments": "summary of traversal arguments if any",
    "linking_claims": [7, 8]
}}"""


# Task 7: Notice of Allowance Analysis
ALLOWANCE_ANALYSIS_PROMPT = """Analyze this Notice of Allowance document.

Document Text:
{document_text}

Extract:
1. The allowed claims
2. Any claim renumbering (application number to patent number)
   CRITICAL: Patent offices renumber allowed claims sequentially. If application claims
   1, 27, and 66 are allowed, they become issued claims 1, 2, and 3 respectively.
   Look for explicit renumbering tables, or infer from the order of allowed claims.
   Also look for Examiner's Amendments that may indicate renumbering.
3. Reasons for Allowance (if stated) — extract the FULL text including specific features
4. Any examiner's amendments
5. Any conditions for allowance

Respond with JSON:
{{
    "allowed_claims": [1, 2, 3, 4, 5],
    "renumbering": [
        {{
            "application_claim_number": 21,
            "issued_claim_number": 1
        }}
    ],
    "reasons_for_allowance": {{
        "stated": true/false,
        "text": "full text of reasons for allowance",
        "key_distinguishing_features": ["feature A", "feature B"],
        "closest_prior_art": ["Smith reference"]
    }},
    "examiner_amendments": [
        {{
            "claim_number": 1,
            "amendment_text": "what was changed"
        }}
    ],
    "conditions": ["any conditions noted"]
}}"""


# Task 8: Terminal Disclaimer Detection
TERMINAL_DISCLAIMER_PROMPT = """Analyze this document for terminal disclaimer information.

Document Text:
{document_text}

Respond with JSON:
{{
    "has_terminal_disclaimer": true/false,
    "disclaimed_patents": [
        {{
            "patent_number": "US X,XXX,XXX",
            "application_number": "if applicable"
        }}
    ],
    "disclaimer_date": "date if available",
    "reason": "obvousness-type double patenting over X"
}}"""


# Task 9: Means-Plus-Function Detection
MEANS_PLUS_FUNCTION_PROMPT = """Analyze the claims and prosecution history for means-plus-function (35 USC 112(f)) issues.

Document Text:
{document_text}

Current Claims:
{claims_text}

Identify:
1. Any claims or elements invoking 112(f)
2. Whether the examiner has construed any elements under 112(f)
3. The corresponding structure in the specification (if identified)

Respond with JSON:
{{
    "means_plus_function_elements": [
        {{
            "claim_number": 1,
            "element_text": "means for processing...",
            "examiner_invoked_112f": true/false,
            "corresponding_structure": "processor configured to...",
            "specification_support": "paragraph numbers or description"
        }}
    ],
    "examiner_statements_on_112f": [
        {{
            "statement": "exact quote",
            "interpretation": "examiner's interpretation"
        }}
    ]
}}"""


# Task 10: Causality Linking
CAUSALITY_LINKING_PROMPT = """Link this amendment/response to the specific rejection it addresses.

Previous Office Action Rejections:
{rejections}

Amendment/Response Document:
{document_text}

For each amendment made, identify which rejection it was responding to.

Respond with JSON:
{{
    "response_links": [
        {{
            "claim_number": 1,
            "rejection_responded_to": {{
                "rejection_id": "from previous analysis",
                "statutory_basis": "103",
                "prior_art": "Smith"
            }},
            "response_type": "amendment/argument/both",
            "amendment_summary": "added limitation X to overcome",
            "argument_summary": "argued distinction based on Y",
            "overcomes_rejection": true/false
        }}
    ]
}}"""


# Task 11: Priority Support Check
PRIORITY_SUPPORT_PROMPT = """Check if newly added claim elements have support in the priority/provisional filing.

New Elements Added in Amendment:
{new_elements}

Provisional/Priority Document Text:
{priority_text}

For each new element, determine if it has written description support.

Respond with JSON:
{{
    "priority_analysis": [
        {{
            "element": "the new limitation",
            "support_status": "supported/unsupported/partial",
            "supporting_text": "quote from priority doc if found",
            "analysis": "explanation of support determination"
        }}
    ]
}}"""


# Task 12: Interview Summary Analysis
INTERVIEW_SUMMARY_PROMPT = """Analyze this Interview Summary for important concessions or agreements.

Document Text:
{document_text}

Extract any statements that could affect claim scope or create prosecution history estoppel.

CRITICAL: Pay special attention to:
1. **Examiner Suggestions** — specific claim amendments the examiner proposed. These are 
   extremely important because if the applicant adopts them, the examiner has effectively 
   defined the claim scope. Extract the EXACT suggested amendment text.
2. **Proposed Amendments** — specific changes discussed that the applicant agreed to make.
   These create strong estoppel because they were negotiated with the examiner.
3. **Agreements on Claim Construction** — any shared understanding of what terms mean.

Respond with JSON:
{{
    "interview_date": "date if available",
    "participants": ["names"],
    "topics_discussed": ["claim 1 limitations", "prior art"],
    "agreements_reached": [
        {{
            "topic": "what was agreed",
            "claims_affected": [1, 2],
            "potential_estoppel": true/false
        }}
    ],
    "proposed_amendments": ["EXACT description of proposed changes, as specific as possible"],
    "examiner_suggestions": ["EXACT text of examiner's suggested claim language or amendment — be as specific as possible, quoting the examiner's words"],
    "examiner_claim_constructions": ["any terms the examiner interpreted or defined during the interview"]
}}"""


# Comprehensive Document Analysis (combines multiple tasks)
COMPREHENSIVE_ANALYSIS_PROMPT = """Perform a comprehensive analysis of this patent prosecution document.

Provided Document Type: {document_type}
Document Text:
{document_text}

Analyze and extract ALL relevant information.
CRITICAL: Identify the specific document type and date from the text. The provided type may be generic.

1. Correct Document Type (e.g., "Non-Final Rejection", "Amendment after Final")
2. Document Date (Mailing or Filing Date)
3. Claims (if present) - full text and dependency structure
4. Rejections/Objections (if Office Action)
5. Arguments and definitions (if Amendment/Response)
6. Restriction information (if Restriction Requirement)
7. Allowance information (if Notice of Allowance)
8. Any terminal disclaimers
9. Any 112(f) means-plus-function issues
10. Any significant statements by Applicant or Examiner

Respond with a comprehensive JSON object:
{{
    "document_type": "The specific document type identified in the text",
    "document_date": "YYYY-MM-DD",
    "claims": {{
        "present": true/false,
        "claim_list": [/* array of claims if present */]
    }},
    "rejections": {{
        "present": true/false,
        "is_final": true/false,
        "rejection_list": [/* array of rejections if present */]
    }},
    "arguments": {{
        "present": true/false,
        "statement_list": [/* array of statements if present */]
    }},
    "restriction": {{
        "present": true/false,
        "details": {{/* restriction details if present */}}
    }},
    "allowance": {{
        "present": true/false,
        "details": {{/* allowance details if present */}}
    }},
    "terminal_disclaimer": {{
        "present": true/false,
        "details": {{/* TD details if present */}}
    }},
    "means_plus_function": {{
        "present": true/false,
        "elements": [/* MPF elements if present */]
    }},
    "key_statements": [
        {{
            "speaker": "Applicant/Examiner",
            "text": "exact quote",
            "significance": "why this matters",
            "estoppel_risk": "high/medium/low"
        }}
    ],
    "summary": "2-3 sentence summary of document significance"
}}"""


# =============================================================================
# POST-PROCESSING PROMPTS (Layer 2/3 — run after all extraction is complete)
# =============================================================================

# Task 13: The Shadow Examiner (Validity Critique)
SHADOW_EXAMINER_PROMPT = """ROLE: You are a hostile patent litigator attempting to invalidate the following allowed claims.
Do not summarize the claims. Critique them.

CLAIMS TO ANALYZE:
{claims_text}

PROSECUTION CONTEXT (amendments and arguments made during prosecution):
{prosecution_context}

STEP 0 — MANDATORY PRE-CHECK (Mixed Statutory Class / IPXL Holdings):
Before any other analysis, perform this check for EVERY claim:
1. Read the preamble of each claim. If it recites a "system", "apparatus", "device", "computer", 
   "server", "processor", or any other structural/hardware term, it is a system/apparatus claim.
2. For every system/apparatus claim identified, scan the ENTIRE claim body for phrases where a 
   HUMAN USER performs an action. Look for patterns like:
   - "the user specifies...", "the user selects...", "the user matches..."
   - "allowing the user to...", "enabling a user to..."
   - "a user inputting...", "receiving user input..."
   - "the user navigates...", "the user views..."
   - Any phrase with "user" + active verb
3. List EVERY such user-action phrase found in each system/apparatus claim.
4. If ANY user-action phrases exist in a system/apparatus claim, report a "Mixed Statutory Class" 
   risk citing IPXL Holdings v. Amazon.com (Fed. Cir. 2005). This is an indefiniteness issue 
   under 35 USC 112(b) because system claims must recite structure, not actions performed by users.

After completing Step 0, proceed to the remaining checklist items:

CHECKLIST:
1. Indefiniteness (35 USC 112(b)): Look for antecedent basis issues ("the element" without prior introduction), subjective terms without clear boundaries ("substantially", "aesthetically pleasing", "about"), or terms that have been inconsistently defined during prosecution.
2. Abstract Idea (35 USC 101): Is the claim purely mathematical, a mental process, or a method of organizing human activity without a concrete technical improvement or hardware tie-in?
3. Written Description / Enablement (35 USC 112(a)): Did amendments add limitations that may lack support in the original specification?
4. Prosecution History Estoppel Vulnerability: Were arguments made during prosecution that narrow the claim scope in ways that could be exploited in a Doctrine of Equivalents analysis?
5. Colloquial / Metaphorical Indefiniteness (Strategic Enhancement): Scan for colloquial or metaphorical action verbs in system/apparatus claims (e.g., "jump", "snap", "hover", "transition", "float", "fly", "bounce", "slide") that describe a visual result without defining the technical mechanism. If the claim says a system is "configured to cause X to jump to Y," what precisely constitutes a "jump" vs. a "transition" vs. a "navigate"? These are indefiniteness targets because they have no established technical meaning in the art. Report these as type "Colloquial Indefiniteness".

IMPORTANT: Only report risks you have HIGH CONFIDENCE about based on the actual claim text. Do not speculate. Each risk must cite the specific claim language that triggers it.

IMPORTANT: For every risk, you MUST include `offending_text_quote` containing the EXACT substring from the claim text that triggers the issue. If you cannot quote specific text, do not report the risk.

Respond with JSON:
{{
    "_reasoning": "STEP 0: First scan each claim preamble for system/apparatus language, then scan body for user-action phrases. Then review each claim for remaining validity issues. For each risk identified, locate the specific claim language that triggers it before filling the fields below.",
    "risks": [
        {{
            "type": "Mixed Statutory Class",
            "claims": [1, 5],
            "severity": "High",
            "offending_text_quote": "the user specifying the approximate location",
            "description": "Claim 1 recites a 'system' (apparatus) but includes method steps performed by a human user: 'the user specifying the approximate location'. Under IPXL Holdings, claims that mix apparatus structure with user method steps are indefinite.",
            "reasoning": "Under IPXL Holdings v. Amazon.com (Fed. Cir. 2005), claims that straddle two statutory classes (system and method) are indefinite under 35 USC 112(b). The claim recites hardware ('a system comprising...') but requires the user to perform actions, making it unclear whether infringement occurs by making/selling the system or by the user's actions."
        }}
    ]
}}"""


# Task 14: Definition Synthesis & Consistency Check (Flip-Flop Detector)
DEFINITION_SYNTHESIS_PROMPT = """ROLE: You are a technical analyst reviewing how the term "{term}" was defined during prosecution.

RAW STATEMENTS (Chronological):
{statements_list}

TASK 1: SYNTHESIS
Write a single, cohesive paragraph defining "{term}" based on these statements.
Do not list them individually. Synthesize the core concept the applicant is conveying.
If the examiner and applicant disagree, note both positions.

CRITICAL ATTRIBUTION RULE: When synthesizing, clearly attribute who holds each position.
If the applicant distinguishes their invention from prior art, the synthesis should reflect
what the APPLICANT'S term means in the context of their invention, not what the prior art teaches.
For example, if the applicant argues "Unlike Reference X which uses dynamic pointing direction,
the claimed invention specifies a static pointing direction," the synthesis for "pointing direction"
should be: "The Applicant defines 'pointing direction' as a static directional attribute..." —
NOT "a dynamic directional attribute" (which is what the prior art teaches, not the invention).

TASK 2: CONSISTENCY CHECK (The "Flip-Flop" Detector)
Did the Applicant change the definition over time to suit their needs?
Examples of flip-flops:
- Arguing the term is "hardware" in 2015 to overcome a 101 rejection, but "software" in 2017 to overcome a 102 rejection.
- Defining a term broadly to capture prior art, then narrowing it to avoid different prior art.
- Contradicting an earlier concession or examiner-acquiesced definition.

If the definition merely evolved or was refined (not contradicted), mark as "Evolving" not "Contradictory".

Respond with JSON:
{{
    "_reasoning": "Read through all statements chronologically. Note whether the core definition stays stable or shifts. Identify any direct contradictions before synthesizing.",
    "term": "{term}",
    "narrative_summary": "The Applicant consistently defines '{term}' as...",
    "consistency_status": "Consistent",
    "contradiction_analysis": "No contradictions found. The definition was refined in [date] to add specificity but remained aligned with the original meaning."
}}"""


# Task 15: Claim Biography / Narrative
CLAIM_NARRATIVE_PROMPT = """ROLE: You are a patent analyst writing the prosecution "biography" of a specific claim.

CLAIM NUMBER: {claim_number}

CLAIM VERSIONS (Chronological):
{versions_text}

REJECTIONS FACED:
{rejections_text}

KEY ARGUMENTS MADE:
{arguments_text}

REASONS FOR ALLOWANCE (if available):
{reasons_for_allowance_text}

IS DEPENDENT CLAIM: {is_dependent}
PARENT CLAIM NUMBER: {parent_claim_number}

TASK:
Write a concise narrative (2-4 sentences) that tells the story of this claim:
- What was the original scope?
- What rejections did it face?
- How was it amended to overcome them?
- What was the key turning point that led to allowance (if allowed)?
- If the Examiner stated Reasons for Allowance, mention the key distinguishing feature the Examiner identified.

CRITICAL TURNING POINT RULE (Gap 13):
The turning point is the amendment or event most directly linked to the examiner's stated 
reasons for allowance — NOT necessarily the largest scope change or the first significant 
amendment. If Reasons for Allowance are available above, identify which amendment added the 
feature the examiner cited as distinguishing. THAT amendment is the turning point.

SPECIAL INSTRUCTIONS FOR DEPENDENT CLAIMS (Gap 6):
If this is a dependent claim:
1. Identify what SPECIFIC ADDITIONAL LIMITATION this dependent claim adds beyond the parent.
2. Check if there are prosecution arguments specifically about this dependent claim's limitation
   (not just inherited from the parent). If there are NONE — the claim was simply allowed because 
   its parent was allowed — then state what the dependent feature is in ONE sentence and stop.
   Set "has_independent_significance" to false.
3. If the dependent limitation WAS independently significant (cited by examiner, used to overcome 
   a rejection, or mentioned in specific arguments), write a full narrative and set 
   "has_independent_significance" to true.

Respond with JSON:
{{
    "claim_number": {claim_number},
    "evolution_summary": "Claim {claim_number} was originally filed as a broad [type] claim covering [scope]. After a [basis] rejection citing [art], the applicant narrowed the claim by adding [limitation]. This was the turning point that led to allowance.",
    "turning_point_event": "Amendment filed YYYY-MM-DD adding '[specific limitation]' to distinguish over [reference]",
    "has_independent_significance": true
}}"""


# Task 16: Key Claim Limitations Synthesis 
KEY_LIMITATIONS_PROMPT = """ROLE: You are a patent litigator preparing a claim construction brief. Analyze the prosecution history to identify the KEY CLAIM LIMITATIONS that were added during prosecution and their legal significance.

FINAL ISSUED CLAIMS:
{claims_text}

PROSECUTION AMENDMENTS (limitations added and arguments made):
{amendments_context}

REASONS FOR ALLOWANCE:
{reasons_for_allowance}

TASK:
For each substantive limitation that was ADDED during prosecution (not present in original filing), produce a structured entry explaining:
1. What the limitation says
2. What prior art it was added to overcome
3. What argument the applicant made
4. The estoppel/litigation consequence
5. How the Examiner or PTAB responded (if known)

Focus on limitations that matter for claim construction and infringement analysis. Skip trivial formatting changes.

Respond with JSON:
{{
    "key_limitations": [
        {{
            "limitation_text": "exact claim language of the limitation",
            "affected_claims": [1, 2],
            "prior_art_overcome": "Reference name(s)",
            "applicant_argument": "The applicant argued that [Reference] did not disclose [X] because...",
            "prosecution_significance": "This language means the claims require [X] and exclude [Y]. The applicant cannot now assert equivalents covering [Y].",
            "examiner_view": "The Examiner agreed / The Examiner's Reasons for Allowance cited this as the distinguishing feature / Not stated"
        }}
    ]
}}"""


# =============================================================================
# GAP ANALYSIS ADDITIONS: New Post-Processing Prompts
# =============================================================================

# Task 17: Strategic Tensions / Contradictions Detection (Gap 1 fix)
STRATEGIC_TENSIONS_PROMPT = """ROLE: You are a patent litigator looking for contradictions and strategic tensions 
in the prosecution history that could be exploited.

AMENDMENTS AND ARGUMENTS (chronological):
{prosecution_events}

TASK:
Review the prosecution history for cases where the applicant argued BOTH SIDES of the same 
feature or concept at different points in prosecution. These are "strategic tensions" — places 
where the applicant:

1. **Argued AGAINST a feature to distinguish one reference, then ADDED that same feature to 
   distinguish a different reference.** 
   Example: "We don't use a geographic map" (to distinguish Reference A) → later adds 
   "two-dimensional geographic map" (to distinguish Reference B).

2. **Defined a term one way to overcome one rejection, then defined it differently to overcome 
   another rejection.**
   Example: Argued "model" means "database" to overcome Ref A, then argued "model" means 
   "3D rendering" to overcome Ref B.

3. **Made a broad categorical exclusion in arguments but then narrowed it in a way that 
   partially contradicts the exclusion.**

4. **Distinguish-then-Adopt Pattern (Pivot Estoppel):** Look for instances where the Applicant 
   distinguished a reference (Ref A) by arguing it LACKED Feature X, but subsequently amended 
   to ADD Feature X to distinguish a DIFFERENT reference (Ref B). This creates a "bootstrapping" 
   irony: the applicant used the ABSENCE of Feature X as a sword (against Ref A) and the 
   PRESENCE of Feature X as a shield (against Ref B). This is a high-value strategic tension 
   because it can be exploited in deposition or claim construction briefing.

For each tension found, explain:
- What was argued first (with the reference being distinguished)
- What was argued/added later (with the reference being distinguished)  
- Why this is a tension (how a litigator could exploit it)

Only report tensions you have HIGH CONFIDENCE about. Do not speculate.

Respond with JSON:
{{
    "_reasoning": "Review the chronological prosecution events, looking for the same concept or feature being argued in opposite directions at different times.",
    "strategic_tensions": [
        {{
            "feature_or_concept": "the feature/concept at the center of the tension (e.g., 'geographic map', 'model definition')",
            "first_position": "What was initially argued and against which reference",
            "first_date": "approximate date of the first position",
            "second_position": "What was later argued/added and against which reference",
            "second_date": "approximate date of the second position",
            "tension_explanation": "Why this is a strategic tension a litigator could exploit",
            "affected_claims": [1, 2],
            "severity": "High/Medium"
        }}
    ]
}}"""


# Task 18: Per-Claim Strategic Implications (Gap 2 fix)
CLAIM_IMPLICATIONS_PROMPT = """ROLE: You are a patent litigator preparing a claim-by-claim strategic assessment.

CLAIM {claim_number} (Issued):
{claim_text}

ESTOPPEL EVENTS AFFECTING THIS CLAIM:
{estoppel_events}

VALIDITY RISKS FOR THIS CLAIM:
{validity_risks}

PROSECUTION ARGUMENTS AFFECTING THIS CLAIM:
{prosecution_arguments}

TASK:
Synthesize a concise strategic implications summary for this claim, covering:

1. **Estoppel Constraints:** What scope has been surrendered? What equivalents are barred?
2. **Indefiniteness Risks:** Any terms that could be challenged as indefinite? Any mixed 
   statutory class (IPXL Holdings) issues?
3. **Categorical Exclusions:** What does this claim categorically exclude based on prosecution arguments?
4. **Directionality Constraints:** Any "A→B not B→A" constraints from prosecution?
5. **Prosecution History Estoppel:** Can the patent holder recapture scope given up during prosecution?

Be specific and cite the actual claim language and prosecution arguments.

Respond with JSON:
{{
    "claim_number": {claim_number},
    "estoppel_constraints": "Summary of what equivalents are barred...",
    "indefiniteness_risks": "Summary of any indefiniteness issues...",
    "categorical_exclusions": ["List of things this claim categorically excludes"],
    "directionality_constraints": ["List of directional requirements"],
    "recapture_analysis": "Whether prosecution history estoppel bars recapture of surrendered scope",
    "overall_vulnerability": "High/Medium/Low — overall assessment of claim vulnerability"
}}"""


# Task 19: Per-Claim Scope Constraints Cheat Sheet (Improvement 2)
SCOPE_CONSTRAINTS_PROMPT = """ROLE: You are creating a quick-reference cheat sheet for a litigator.

CLAIM {claim_number} (Issued):
{claim_text}

PROSECUTION HISTORY SUMMARY:
{prosecution_summary}

TASK:
Create a concise scope constraints summary using these categories:
- REQUIRES: Key affirmative requirements from the claim text and prosecution
- EXCLUDES: Things categorically excluded by prosecution arguments or negative limitations
- ESTOPPEL: Equivalents that cannot be recaptured due to prosecution history estoppel
- DIRECTION: Any directionality constraints (A→B, not B→A)

Be specific. Use actual claim language and prosecution arguments.

Respond with JSON:
{{
    "claim_number": {claim_number},
    "requires": ["list of affirmative requirements"],
    "excludes": ["list of categorical exclusions"],
    "estoppel_bars": ["list of equivalents barred by estoppel"],
    "directionality": ["list of directional constraints"]
}}"""


# =============================================================================
# Task 20: Thematic Synthesis (Improvement A)
# Groups amendments and arguments into overarching "Themes of Patentability"
# =============================================================================
THEMATIC_SYNTHESIS_PROMPT = """ROLE: You are a patent litigator preparing a thematic summary of the prosecution history.

ALL PROSECUTION ARGUMENTS (chronological):
{arguments_text}

ALL AMENDMENTS / ADDED LIMITATIONS (chronological):
{amendments_text}

REASONS FOR ALLOWANCE (if available):
{reasons_for_allowance}

TASK:
Analyze the collection of arguments and amendments above. Group them into 3-7 core 
"Themes of Patentability" — the overarching technical concepts and distinctions the applicant 
used to overcome prior art and secure the patent.

For example, themes might be:
- "Integrity of Original Images" (applicant argued the invention does NOT alter/warp images)
- "Two-Step Location Refinement" (user specifies approximate location, then system refines)
- "Catalog vs. Rendering" (3D model serves as a navigational catalog, not a rendered scene)

For each theme:
1. Give it a short, descriptive title
2. Write a summary (2-4 sentences) explaining the core concept
3. List the key arguments and amendments that support this theme
4. Note which prior art references this theme distinguishes
5. Assess the estoppel significance (what scope is surrendered by this theme)

IMPORTANT: 
- Themes should be CONCEPTUAL, not just "arguments about Reference X"
- Look for METAPHORS, ANALOGIES, and CONCEPTUAL CHARACTERIZATIONS the applicant used
  (e.g., images "hover" in "ether", model acts as a "catalog", "digital filing cabinet")
- Each theme should group MULTIPLE arguments/amendments that share a common conceptual thread
- If the Reasons for Allowance highlight a specific feature, ensure it maps to a theme

TOKEN MANAGEMENT: If you are running out of token space, prioritize finishing the current 
theme's JSON object cleanly over starting a new theme. Complete themes are more useful than 
truncated ones.

Respond with JSON:
{{
    "_reasoning": "Review all arguments and amendments to identify recurring conceptual threads. Group related arguments under thematic umbrellas rather than by document or reference.",
    "themes": [
        {{
            "title": "Short descriptive theme title (e.g., 'Integrity of Original Images')",
            "summary": "2-4 sentence explanation of this theme's core concept and why it matters for patentability",
            "key_arguments": [
                "Brief summary of a supporting argument (with reference name and date if available)"
            ],
            "key_amendments": [
                "Brief summary of an amendment that supports this theme (e.g., 'Added without alteration to Claim 1')"
            ],
            "prior_art_distinguished": ["Reference A", "Reference B"],
            "affected_claims": [1, 2, 3],
            "estoppel_significance": "What scope is surrendered by taking this position (e.g., 'Cannot now argue images may be altered/warped')",
            "metaphors_or_analogies": ["Any metaphors/analogies the applicant used for this theme (e.g., 'images hover in ether')"]
        }}
    ]
}}"""


# =============================================================================
# Task 21: Term Boundary Extraction (Gap 1)
# Extracts specific examples of what falls inside/outside a term's scope
# =============================================================================
TERM_BOUNDARY_EXTRACTION_PROMPT = """ROLE: You are a patent claim construction expert extracting actionable scope boundaries.

TERM: "{term}"

PROSECUTION STATEMENTS ABOUT THIS TERM (chronological):
{statements_list}

TASK:
For each statement above, identify any SPECIFIC EXAMPLES the applicant or examiner gave 
of things that DO or DO NOT fall within the term's scope. Extract each example as a 
separate boundary entry.

Look for patterns like:
- "Without modification" means NO filtering, NO truncating, NO resampling, NO cropping
  → 4 separate "excludes" entries: "filtering", "truncating", "resampling", "cropping"
- "Rigid connection" includes welded joints and bolted flanges
  → 2 separate "includes" entries: "welded joints", "bolted flanges"
- "Real-time" means latency under 100ms
  → 1 "includes" entry: "latency under 100ms"
- "The model serves as a catalog, not a rendered scene"
  → 1 "includes" entry: "catalog/index function" + 1 "excludes" entry: "rendered scene"

Only extract CONCRETE, SPECIFIC examples — not abstract restatements of the term itself.
If no specific examples are enumerated, return an empty list.

Respond with JSON:
{{
    "_reasoning": "Review each statement for specific enumerated examples of what is included or excluded from the term's scope.",
    "term": "{term}",
    "boundaries": [
        {{
            "boundary_type": "includes|excludes",
            "example_text": "The specific enumerated item (e.g., 'filtering', 'welded joints', 'latency under 100ms')",
            "source_quote": "Brief quote from the prosecution statement that establishes this boundary",
            "confidence": "high|medium"
        }}
    ]
}}"""


# =============================================================================
# Task 22: Per-Claim Vulnerability Card (Gap 5)
# Unified per-claim assessment for litigation use
# =============================================================================
VULNERABILITY_CARD_PROMPT = """ROLE: You are a patent litigator preparing a per-claim vulnerability assessment.

CLAIM {claim_number} (Issued Text):
{claim_text}

PROSECUTION STATEMENTS AFFECTING THIS CLAIM:
{prosecution_statements}

VALIDITY RISKS FOR THIS CLAIM:
{validity_risks}

ESTOPPEL EVENTS AFFECTING THIS CLAIM:
{estoppel_events}

AMENDMENT HISTORY:
{amendment_history}

TASK:
Synthesize ALL of the above into a single vulnerability card for Claim {claim_number}.
This card should be directly usable by a litigator for infringement analysis and design-around planning.

For each category, be SPECIFIC — cite actual claim language and prosecution arguments.

Respond with JSON:
{{
    "claim_number": {claim_number},
    "must_practice": [
        "Specific affirmative requirement from claim text or prosecution (e.g., 'Must display a two-dimensional geographic map')"
    ],
    "categorical_exclusions": [
        "Things categorically excluded by prosecution arguments (e.g., 'Cannot modify/alter/crop the 2D images')"
    ],
    "estoppel_bars": [
        "Equivalents foreclosed by prosecution history (e.g., 'Cannot recapture scope over non-geographic maps — surrendered to distinguish Kato')"
    ],
    "indefiniteness_targets": [
        "Specific claim language vulnerable to challenge (e.g., 'User specifying approximate location — mixed method/apparatus issue')"
    ],
    "design_around_paths": [
        "Suggested escape routes (e.g., 'Use automated location detection instead of user specification')"
    ],
    "overall_vulnerability": "High|Medium|Low"
}}"""