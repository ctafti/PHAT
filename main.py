#!/usr/bin/env python3
"""
PHAT - Patent Prosecution History Analysis Tool v2.0
Main Entry Point

Usage:
    python main.py                      # Process all PDFs in input folder
    python main.py --file <path>        # Process a specific PDF
    python main.py --report <patent_id> # Generate report for analyzed patent
"""

import sys
import os
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime

import yaml

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import DatabaseManager, Patent
from src.ai_providers import AIProviderFactory
from src.pdf_processor import PDFProcessor
from src.analysis_engine import AnalysisEngine
from src.report_generator import ReportGenerator
from src.final_claims import (
    prompt_for_final_claims,
    parse_google_patents_claims,
    fetch_claims_from_google_patents
)


def setup_logging(config: dict):
    """Configure logging based on config"""
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    log_file = config.get('logging', {}).get('file', 'output/phat.log')
    
    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure required directories exist
    for path_key in ['input_folder', 'output_folder']:
        path = config['paths'].get(path_key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    # Ensure database directory exists
    db_path = config.get('paths', {}).get('database', 'output/phat.db')
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    return config


def find_pdfs(input_folder: str) -> list:
    """Find all PDF files in the input folder"""
    input_path = Path(input_folder)
    return list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF"))


def extract_patent_number_from_pdf(pdf_path: str) -> str:
    """Quick extraction of patent number from PDF for final claims lookup"""
    # 1. Try filename extraction first (fast & robust)
    try:
        filename = Path(pdf_path).name
        # Matches "8,026,929" or "8026929"
        match = re.search(r"(?:US\s*Patent\s*(?:No\.?)?|US)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+)", filename, re.IGNORECASE)
        if match:
            return match.group(1)
    except Exception:
        pass

    # 2. Fallback to PDF content extraction
    try:
        processor = PDFProcessor(pdf_path)
        processor.extract_text()
        info = processor.get_patent_info()
        return info.get('patent_number')
    except Exception:
        return None


def get_final_claims_for_patent(pdf_path: str, config: dict, logger: logging.Logger,
                                 skip_prompt: bool = False, claims_file: str = None) -> tuple:
    """
    Get final claims for a patent - either from file, auto-fetch, prompt, or skip.
    
    Returns:
        tuple: (claims_text, parsed_claims, verified_patent_number) or (None, None, None) if skipped
    """
    verified_patent_number = None
    
    # If claims file is provided, read from it
    if claims_file:
        try:
            with open(claims_file, 'r') as f:
                claims_text = f.read()
            parsed_claims = parse_google_patents_claims(claims_text)
            if parsed_claims:
                logger.info(f"Loaded {len(parsed_claims)} final claims from {claims_file}")
                return claims_text, parsed_claims, None
            else:
                logger.warning(f"Could not parse claims from {claims_file}")
        except Exception as e:
            logger.warning(f"Could not read claims file {claims_file}: {e}")
    
    # If skipping prompt, return None
    if skip_prompt:
        logger.info("Skipping final claims input (--no-claims flag)")
        return None, None, None
    
    # Try to get patent number for Google Patents URL
    patent_number = extract_patent_number_from_pdf(pdf_path)
    
    # Attempt auto-fetch if patent number exists
    if patent_number:
        try:
            logger.info(f"Attempting to auto-fetch claims for {patent_number}...")
            claims_text = fetch_claims_from_google_patents(patent_number)
            if claims_text:
                parsed_claims = parse_google_patents_claims(claims_text)
                if parsed_claims:
                    logger.info(f"Successfully auto-fetched {len(parsed_claims)} claims from Google Patents")
                    # The patent_number used to successfully fetch is the verified issued number
                    verified_patent_number = patent_number
                    return claims_text, parsed_claims, verified_patent_number
        except Exception as e:
            logger.warning(f"Auto-fetch failed: {e}")
    
    # Prompt user for final claims
    claims_text = prompt_for_final_claims(patent_number)
    
    if claims_text:
        parsed_claims = parse_google_patents_claims(claims_text)
        # If user provided claims and we had a patent number, treat it as verified
        if parsed_claims and patent_number:
            verified_patent_number = patent_number
        return claims_text, parsed_claims, verified_patent_number
    
    return None, None, None


def process_single_pdf(pdf_path: str, config: dict, logger: logging.Logger,
                       final_claims_text: str = None,
                       final_claims_parsed: list = None,
                       resume_patent_id: str = None,
                       verified_patent_number: str = None) -> str:
    """Process a single PDF file and return the patent ID.
    
    Args:
        pdf_path: Path to the PDF file
        config: Configuration dictionary
        logger: Logger instance
        final_claims_text: Raw text of final claims (optional)
        final_claims_parsed: Parsed final claims list (optional)
        resume_patent_id: If provided, skip OCR-based patent identification
                          and resume analysis for this known patent ID.
        verified_patent_number: Patent number verified via Google Patents (optional)
    """
    logger.info(f"Processing: {pdf_path}")
    
    try:
        engine = AnalysisEngine(config)
        
        # Pass final claims to the analysis engine
        if final_claims_text and final_claims_parsed:
            engine.set_final_claims(final_claims_text, final_claims_parsed)
        
        # Pass verified patent number if available
        if verified_patent_number:
            engine.set_verified_patent_number(verified_patent_number)
        
        # Pass resume_patent_id to skip OCR-based identification
        patent_id = engine.analyze_file_history(pdf_path, resume_patent_id=resume_patent_id)
        logger.info(f"Analysis complete. Patent ID: {patent_id}")
        return patent_id
        
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        raise


def generate_report(patent_id: str, config: dict, logger: logging.Logger) -> dict:
    """Generate report for a processed patent"""
    logger.info(f"Generating report for patent: {patent_id}")
    
    try:
        db_manager = DatabaseManager(config['paths']['database'])
        report_gen = ReportGenerator(db_manager, config['paths']['output_folder'])
        
        report_format = config.get('report', {}).get('format', 'both')
        output_files = report_gen.generate_report(patent_id, format=report_format)
        
        logger.info(f"Report generated: {output_files}")
        return output_files
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise


def add_final_claims_to_existing(patent_id: str, config: dict, logger: logging.Logger) -> bool:
    """Add final claims to an existing patent analysis"""
    db_manager = DatabaseManager(config['paths']['database'])
    session = db_manager.get_session()
    
    try:
        patent = session.query(Patent).get(patent_id)
        if not patent:
            print(f"Error: Patent ID '{patent_id}' not found")
            return False
        
        print(f"\nPatent: {patent.patent_number or patent.application_number or patent_id}")
        
        claims_text = prompt_for_final_claims(patent.patent_number)
        
        if claims_text:
            parsed_claims = parse_google_patents_claims(claims_text)
            if parsed_claims:
                db_manager.store_final_claims(patent_id, claims_text, parsed_claims)
                print(f"✓ Stored {len(parsed_claims)} final claims")
                return True
            else:
                print("Could not parse claims from input")
                return False
        else:
            print("No claims provided")
            return False
            
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(
        description="PHAT - Patent Prosecution History Analysis Tool v2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                           # Process all PDFs in input/
    python main.py --file patent.pdf         # Process specific PDF
    python main.py --fast                    # Use fast/cheap model for testing
    python main.py --report abc123           # Generate report for patent ID
    python main.py --provider gemini --fast  # Use Gemini Flash
    python main.py --list                    # List analyzed patents
    python main.py --no-claims               # Skip final claims prompt
    python main.py --claims-file claims.txt  # Load claims from file
    python main.py --add-claims abc123       # Add claims to existing analysis
        """
    )
    
    parser.add_argument('--file', '-f', type=str, help='Path to specific PDF to process')
    parser.add_argument('--report', '-r', type=str, help='Generate report for patent ID')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--provider', '-p', type=str, choices=['claude', 'gemini', 'deepseek'],
                        help='Override AI provider from config')
    parser.add_argument('--fast', action='store_true',
                        help='Use fast/cheap model for testing (e.g., Haiku, Flash)')
    parser.add_argument('--list', '-l', action='store_true', help='List analyzed patents')
    parser.add_argument('--no-report', action='store_true', help='Skip report generation after analysis')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    # Final claims arguments
    parser.add_argument('--no-claims', action='store_true',
                        help='Skip the final claims input prompt')
    parser.add_argument('--claims-file', type=str,
                        help='Path to text file containing final claims')
    parser.add_argument('--add-claims', type=str, metavar='PATENT_ID',
                        help='Add final claims to an existing patent analysis')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        print("Please ensure config.yaml exists in the project directory.")
        sys.exit(1)
    
    # Override provider if specified
    if args.provider:
        config['ai_provider'] = args.provider
    
    # Use fast model if --fast flag specified (overrides config)
    if args.fast:
        config['model_mode'] = 'fast'
    
    # Override verbose if specified
    if args.verbose:
        config.setdefault('logging', {})['verbose'] = True
    
    # Setup logging
    logger = setup_logging(config)
    
    # Get model info for display
    model_info = AIProviderFactory.get_model_info(config)
    verbose_mode = config.get('logging', {}).get('verbose', False)
    
    logger.info("=" * 60)
    logger.info("PHAT - Patent Prosecution History Analysis Tool v2.1")
    logger.info(f"AI Provider: {model_info['provider']}")
    logger.info(f"Model: {model_info['model']} ({model_info['mode']} mode)")
    logger.info(f"PDF Strategy: bookmarks-first (with pattern fallback)")
    logger.info(f"Post-Processing: {config.get('post_processing', {}).get('enabled', True)}")
    logger.info(f"Verbose Logging: {'ENABLED' if verbose_mode else 'disabled'}")
    if verbose_mode:
        logger.info("WARNING: Verbose mode creates large log files!")
    logger.info("=" * 60)
    
    # Add claims to existing analysis mode
    if args.add_claims:
        success = add_final_claims_to_existing(args.add_claims, config, logger)
        if success:
            # Regenerate report
            print("\nRegenerating report with final claims...")
            output_files = generate_report(args.add_claims, config, logger)
            print("Report updated:")
            for fmt, path in output_files.items():
                print(f"  {fmt}: {path}")
        sys.exit(0 if success else 1)
    
    # List mode
    if args.list:
        db_manager = DatabaseManager(config['paths']['database'])
        session = db_manager.get_session()
        patents = session.query(Patent).all()
        
        if patents:
            print("\nAnalyzed Patents:")
            print("-" * 70)
            for p in patents:
                claims_status = "✓ Has final claims" if p.has_final_claims else "✗ No final claims"
                doc_count = len(p.documents) if hasattr(p, 'documents') else 'N/A'
                print(f"  ID: {p.id}")
                print(f"  Patent Number: {p.patent_number or 'N/A'}")
                print(f"  Application: {p.application_number or 'N/A'}")
                print(f"  Title: {p.title or 'N/A'}")
                print(f"  Documents: {doc_count}")
                print(f"  Final Claims: {claims_status}")
                print(f"  Analyzed: {p.created_at}")
                print("-" * 70)
        else:
            print("\nNo patents have been analyzed yet.")
        
        session.close()
        return
    
    # Report generation mode
    if args.report:
        try:
            output_files = generate_report(args.report, config, logger)
            print("\nReport generated successfully!")
            for fmt, path in output_files.items():
                print(f"  {fmt}: {path}")
        except Exception as e:
            print(f"\nError generating report: {e}")
            sys.exit(1)
        return
    
    # =========================================================================
    # PROCESS MODE - with filename pre-check for skip/resume
    # =========================================================================
    pdfs_to_process = []
    
    if args.file:
        pdf_path = Path(args.file)
        if not pdf_path.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        pdfs_to_process = [pdf_path]
    else:
        pdfs_to_process = find_pdfs(config['paths']['input_folder'])
    
    if not pdfs_to_process:
        print(f"\nNo PDF files found in {config['paths']['input_folder']}/")
        print("Please place patent file history PDFs in the input folder.")
        sys.exit(0)
    
    print(f"\nFound {len(pdfs_to_process)} PDF(s) to process:")
    for pdf in pdfs_to_process:
        print(f"  - {pdf.name}")
    print()
    
    # Initialize DB manager once for pre-checks
    db_manager = DatabaseManager(config['paths']['database'])
    
    # Process each PDF (with filename pre-check)
    results = []
    for pdf_path in pdfs_to_process:
        try:
            filename = pdf_path.name
            
            # =============================================================
            # FILENAME PRE-CHECK
            # Skip completed files instantly, resume failed/running files
            # without relying on OCR extraction of patent info.
            # =============================================================
            existing_id, status = db_manager.get_patent_status_by_filename(filename)
            
            if status == 'completed' and not args.report:
                print(f"⏭ Skipping {filename} (Status: Completed)")
                print(f"  Use --report {existing_id} to regenerate the report.")
                results.append({
                    'pdf': str(pdf_path),
                    'patent_id': existing_id,
                    'status': 'skipped',
                })
                continue
            
            resume_id = None
            if existing_id:
                print(f"🔄 Resuming analysis for {filename} (Previous status: {status})")
                resume_id = existing_id
            # =============================================================
            
            # Get final claims for this PDF
            claims_text, claims_parsed, verified_patent_number = get_final_claims_for_patent(
                str(pdf_path), config, logger,
                skip_prompt=args.no_claims,
                claims_file=args.claims_file
            )
            
            if claims_parsed:
                print(f"\n✓ Using {len(claims_parsed)} final claims as ground truth")
            else:
                print("\n⚠ Proceeding without final claims (analysis will be less accurate)")
            
            # Process with resume support
            patent_id = process_single_pdf(
                str(pdf_path), config, logger,
                final_claims_text=claims_text,
                final_claims_parsed=claims_parsed,
                resume_patent_id=resume_id,
                verified_patent_number=verified_patent_number,
            )
            
            # Generate report unless skipped
            if not args.no_report:
                output_files = generate_report(patent_id, config, logger)
                results.append({
                    'pdf': str(pdf_path),
                    'patent_id': patent_id,
                    'reports': output_files,
                    'has_final_claims': claims_parsed is not None,
                    'status': 'success'
                })
            else:
                results.append({
                    'pdf': str(pdf_path),
                    'patent_id': patent_id,
                    'has_final_claims': claims_parsed is not None,
                    'status': 'success'
                })
                
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            results.append({
                'pdf': str(pdf_path),
                'error': str(e),
                'status': 'failed'
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r['status'] == 'success']
    skipped = [r for r in results if r['status'] == 'skipped']
    failed = [r for r in results if r['status'] == 'failed']
    
    if skipped:
        print(f"\nSkipped (already completed): {len(skipped)}")
        for r in skipped:
            print(f"  ⏭ {Path(r['pdf']).name} (ID: {r['patent_id']})")
    
    print(f"\nSuccessful: {len(successful)}")
    for r in successful:
        print(f"  ✓ {Path(r['pdf']).name}")
        if 'reports' in r:
            for fmt, path in r['reports'].items():
                print(f"      {fmt}: {path}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  ✗ {Path(r['pdf']).name}: {r['error']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()