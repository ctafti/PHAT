"""
PHAT Final Claims Handler
Handles parsing and storing final claims from Google Patents
"""

import re
import sys
import logging
import html
import httpx
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class FinalClaim:
    """Represents a final issued claim from Google Patents"""
    number: int
    text: str
    is_independent: bool
    depends_on: Optional[int] = None


def fetch_claims_from_google_patents(patent_number: str) -> Optional[str]:
    """
    Automatically fetch claims text from Google Patents.
    
    Args:
        patent_number: The patent number string (e.g., "10,123,456")
        
    Returns:
        The extracted claims text or None if failed
    """
    if not patent_number:
        return None
        
    try:
        # Clean patent number: remove commas, spaces, etc.
        clean_number = re.sub(r'[^\w]', '', patent_number)
        
        # Ensure 'US' prefix if missing and it looks like a number
        if not clean_number.upper().startswith('US') and clean_number[0].isdigit():
            clean_number = f"US{clean_number}"
            
        url = f"https://patents.google.com/patent/{clean_number}/en"
        logger.info(f"Attempting to fetch claims from: {url}")
        
        # Request with timeout and redirects
        response = httpx.get(url, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
        
        # Extract claims section using regex for 'itemprop="claims"'
        # This targets the specific section containing claims text
        match = re.search(
            r'<section[^>]*itemprop=["\']claims["\'][^>]*>(.*?)</section>', 
            response.text, 
            re.DOTALL | re.IGNORECASE
        )
        
        if match:
            claims_html = match.group(1)
            
            # Convert HTML to plain text
            # 1. Replace block elements with newlines
            text = re.sub(r'<(br|div|p|li)[^>]*>', '\n', claims_html, flags=re.IGNORECASE)
            # 2. Remove all remaining tags
            text = re.sub(r'<[^>]+>', '', text)
            # 3. Decode HTML entities (e.g., &lt; -> <)
            text = html.unescape(text)
            # 4. Collapse multiple newlines
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            return text.strip()
            
        logger.warning("Could not identify claims section in Google Patents page")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to fetch from Google Patents: {e}")
        return None


def parse_google_patents_claims(claims_text: str) -> List[FinalClaim]:
    """
    Parse claims text copied from Google Patents.
    
    Expected format:
    1. A system for... comprising:
       element a;
       element b.
    2. The system of claim 1, wherein...
    3. A method for... comprising:
       step a;
       step b.
    ...
    
    Returns list of FinalClaim objects.
    """
    claims = []
    
    if not claims_text or not claims_text.strip():
        return claims
    
    # Normalize line endings and clean up
    text = claims_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Pattern to match claim numbers at start of line or after newline
    # Handles: "1. A system...", "\n1. A system...", "  1. A system..."
    claim_pattern = r'(?:^|\n)\s*(\d+)\.\s+'
    
    # Find all claim start positions
    matches = list(re.finditer(claim_pattern, text))
    
    if not matches:
        logger.warning("No claims found in the provided text")
        return claims
    
    for i, match in enumerate(matches):
        claim_num = int(match.group(1))
        start_pos = match.end()
        
        # End position is either the start of next claim or end of text
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)
        
        claim_text = text[start_pos:end_pos].strip()
        
        # Determine if independent or dependent
        depends_on = None
        is_independent = True
        
        # Check for dependency patterns
        dep_patterns = [
            r'(?:The\s+)?(?:system|method|apparatus|device|medium|computer|processor|machine|article|product)\s+(?:of|according\s+to|as\s+(?:set\s+forth\s+|recited\s+|claimed\s+)?in)\s+claim\s+(\d+)',
            r'claim\s+(\d+)\s*,\s*(?:wherein|where|further|additionally)',
            r'(?:as\s+)?(?:recited|set\s+forth|claimed)\s+in\s+claim\s+(\d+)',
        ]
        
        for pattern in dep_patterns:
            dep_match = re.search(pattern, claim_text, re.IGNORECASE)
            if dep_match:
                depends_on = int(dep_match.group(1))
                is_independent = False
                break
        
        claims.append(FinalClaim(
            number=claim_num,
            text=claim_text,
            is_independent=is_independent,
            depends_on=depends_on
        ))
    
    logger.info(f"Parsed {len(claims)} final claims ({sum(1 for c in claims if c.is_independent)} independent)")
    return claims


def prompt_for_final_claims(patent_number: str = None) -> Optional[str]:
    """
    Interactive prompt to get final claims from user.
    
    Instructs user to go to Google Patents and paste the claims.
    Returns the pasted text or None if skipped.
    """
    print("\n" + "=" * 70)
    print("FINAL CLAIMS INPUT")
    print("=" * 70)
    
    if patent_number:
        google_url = f"https://patents.google.com/patent/US{patent_number.replace(',', '')}"
        print(f"\nTo get the final claims for this patent:")
        print(f"  1. Go to: {google_url}")
        print(f"  2. Scroll to the 'Claims' section")
        print(f"  3. Copy all claims text (including claim numbers)")
        print(f"  4. Paste below and press Enter twice when done")
    else:
        print("\nTo get the final claims for this patent:")
        print("  1. Go to Google Patents (https://patents.google.com)")
        print("  2. Search for the patent by number")
        print("  3. Scroll to the 'Claims' section")
        print("  4. Copy all claims text (including claim numbers)")
        print("  5. Paste below and press Enter twice when done")
    
    print("\nPaste claims below (or type 'skip' to continue without final claims):")
    print("-" * 70)
    
    lines = []
    empty_count = 0
    
    try:
        while True:
            try:
                line = input()
            except EOFError:
                break
            
            if line.strip().lower() == 'skip':
                print("\nSkipping final claims input.")
                return None
            
            if line.strip() == '':
                empty_count += 1
                if empty_count >= 2:
                    break
                lines.append(line)
            else:
                empty_count = 0
                lines.append(line)
    except KeyboardInterrupt:
        print("\n\nInput cancelled.")
        return None
    
    claims_text = '\n'.join(lines).strip()
    
    if not claims_text:
        print("\nNo claims provided. Continuing without final claims.")
        return None
    
    # Quick validation
    claims = parse_google_patents_claims(claims_text)
    if claims:
        print(f"\n✓ Successfully parsed {len(claims)} claims")
        print(f"  - Independent claims: {[c.number for c in claims if c.is_independent]}")
        print(f"  - Dependent claims: {[c.number for c in claims if not c.is_independent]}")
    else:
        print("\n⚠ Warning: Could not parse any claims from the input.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return None
    
    print("-" * 70)
    return claims_text


def format_claims_for_display(claims: List[FinalClaim], max_text_length: int = 200) -> str:
    """Format claims list for display in reports"""
    output = []
    
    for claim in claims:
        claim_type = "Independent" if claim.is_independent else f"Depends on claim {claim.depends_on}"
        text_preview = claim.text[:max_text_length]
        if len(claim.text) > max_text_length:
            text_preview += "..."
        
        output.append(f"**Claim {claim.number}** ({claim_type})\n{text_preview}")
    
    return "\n\n".join(output)


def compare_claim_texts(prosecution_text: str, final_text: str) -> Dict[str, Any]:
    """
    Compare prosecution claim text with final issued claim text.
    
    Returns dict with:
    - is_match: True if texts are substantially similar
    - similarity_score: 0.0 to 1.0
    - differences: List of identified differences
    """
    def normalize(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ''
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    norm_pros = normalize(prosecution_text)
    norm_final = normalize(final_text)
    
    # Simple word-based similarity
    pros_words = set(norm_pros.split())
    final_words = set(norm_final.split())
    
    if not pros_words or not final_words:
        return {
            'is_match': False,
            'similarity_score': 0.0,
            'differences': ['One or both texts are empty']
        }
    
    intersection = pros_words & final_words
    union = pros_words | final_words
    
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # Words in final but not in prosecution (additions)
    additions = final_words - pros_words
    # Words in prosecution but not in final (removals)
    removals = pros_words - final_words
    
    differences = []
    if additions:
        differences.append(f"Added terms: {', '.join(list(additions)[:10])}")
    if removals:
        differences.append(f"Removed terms: {', '.join(list(removals)[:10])}")
    
    return {
        'is_match': jaccard > 0.85,
        'similarity_score': jaccard,
        'differences': differences
    }


def create_claim_mapping(prosecution_claims: List[Dict], final_claims: List[FinalClaim]) -> Dict[int, int]:
    """
    Attempt to map prosecution claim numbers to final claim numbers.
    
    Uses text similarity to find best matches.
    Returns dict mapping prosecution_claim_num -> final_claim_num
    """
    mapping = {}
    
    # First pass: try exact number matches
    final_by_num = {c.number: c for c in final_claims}
    
    for pros_claim in prosecution_claims:
        pros_num = pros_claim.get('application_number') or pros_claim.get('number')
        
        if pros_num in final_by_num:
            # Check if texts are similar enough
            pros_text = pros_claim.get('text', '') or ''
            final_text = final_by_num[pros_num].text
            
            comparison = compare_claim_texts(pros_text, final_text)
            if comparison['similarity_score'] > 0.5:
                mapping[pros_num] = pros_num
    
    # Second pass: for unmatched prosecution claims, try to find similar final claims
    unmatched_pros = [c for c in prosecution_claims 
                      if (c.get('application_number') or c.get('number')) not in mapping]
    unmatched_final = [c for c in final_claims if c.number not in mapping.values()]
    
    for pros_claim in unmatched_pros:
        pros_num = pros_claim.get('application_number') or pros_claim.get('number')
        pros_text = pros_claim.get('text', '') or ''
        
        if not pros_text:
            continue
        
        best_match = None
        best_score = 0.5  # Minimum threshold
        
        for final_claim in unmatched_final:
            comparison = compare_claim_texts(pros_text, final_claim.text)
            if comparison['similarity_score'] > best_score:
                best_score = comparison['similarity_score']
                best_match = final_claim.number
        
        if best_match:
            mapping[pros_num] = best_match
            unmatched_final = [c for c in unmatched_final if c.number != best_match]
    
    return mapping


if __name__ == "__main__":
    # Test the parser with sample input
    sample_claims = """
1. A system for generating pose data for posing a 2D image of a scene aligned within the environment of the scene as rendered by a 3D model of the environment of the scene and of numerous other scenes, the system comprising:
a user interface configured to receive a digital representation of the 2D image;
a user interface configured to allow the user to specify an approximate location of the scene on a two-dimensional geographic map;
a 3D graphics engine configured to cause the 3D model to jump to a rendering at the approximate location specified by the user in response to the user specifying the approximate location so as to place corresponding features common to both the 2D image and the jumped-to-rendering of the 3D model simultaneously in view of the user in a manner sufficient to allow the user to match these corresponding features;
a user interface configured to allow the user to match the corresponding features between the 2D image and the rendering of the 3D model; and
a processing system configured to generate the pose data based on the corresponding matched between the 2D image and the rendering of the 3D model by the user.
2. A non-transitory program storage medium, readable by a computer system, embodying a program of instructions executable by the computer system to perform a method for generating pose data for posing a 2D image of a scene aligned within the environment of the scene as rendered by a 3D model of the environment of the scene and of numerous other scenes, the instructions comprising code for:
receiving a digital representation of the 2D image supplied by a user;
allowing the user to specify the approximate location of the scene on a two-dimensional map;
causing the 3D model to jump to a rendering at the approximate location specified by the user in response to the user specifying the approximate location so as to place corresponding features common to both the 2D image and the jumped-to-rendering of the 3D model simultaneously in view of the user in a manner sufficient to allow the user to match these corresponding features;
allowing the user to match the corresponding features between the 2D image and the rendering of the 3D model;
and generating the pose data based on the corresponding features matched between the 2D image and the rendering of the 3D model by the user.
3. A system for allowing a user to freely navigate through a 3D model to view 2D images of scenes posed within the 3D model, the system comprising:
a computer storage system containing:
a plurality of 2D images, each of a scene; and
a pre-existing 3D model which includes the environment of each scene and which is not constructed using the 2D images;
a computer system configured to allow a user to:
navigate to different locations throughout the 3D model;
to see a visual indicator of each 2D image within the 3D model at a location within the 3D model that corresponds to the location of the scene depicted by the 2D image that changes in viewpoint as the user navigates through the 3D model;
to take action in connection with each visual indicator; and
in response to the user taking action with respect to a visual indicator, to cause the rendering of the 3D model to snap to the pose of the scene depicted by the 2D image which corresponds to the visual indicator and to cause the user to see the 2D image posed as a substantially-aligned overlay within the 3D model in a manner which respects the integrity of the original 2D images by presenting them without alteration;
a display system configured to display the 3D model, the visual indicators, and the 2D images posed as substantially-aligned overlays within the pre-existing 3D model in a manner which respects the integrity of the original 2D images by presenting them without alteration; and
a user interface configured to allow the user to navigate through the 3D model and to see the visual indicators and the 2D images posed as substantially-aligned overlays within the 3D model in a manner which respects the integrity of the original 2D images by presenting them without alteration.
4. The system of claim 3 wherein the pre-existing 3D model is of an area of the earth.
5. The system of claim 3 wherein the computer system is configured to receive an upload of the 2D images from the user.
"""

    claims = parse_google_patents_claims(sample_claims)
    
    print(f"Parsed {len(claims)} claims:")
    for claim in claims:
        print(f"\nClaim {claim.number} ({'Independent' if claim.is_independent else f'Depends on {claim.depends_on}'}):")
        print(f"  {claim.text[:100]}...")