#!/usr/bin/env python3
"""
Main entry point for the Crew Feedback Parser System.

This script processes crew feedback forms (PDF, PNG, JPG, TIFF) from a specified folder
and extracts structured data to an Excel file using the LlamaIndex API.

Usage:
    python main.py input_folder output_file.xlsx [options]

Examples:
    # Basic usage
    python main.py ./feedback_forms ./output/crew_feedback.xlsx
    
    # With verbose logging
    python main.py ./feedback_forms ./output/crew_feedback.xlsx --verbose
    
    # Process single file
    python main.py ./feedback_forms ./output/crew_feedback.xlsx --single-file form1.pdf
    
    # With custom worker count
    python main.py ./feedback_forms ./output/crew_feedback.xlsx --workers 5
    
    # Generate processing report
    python main.py ./feedback_forms ./output/crew_feedback.xlsx --report-file ./reports/processing_report.json

Environment Variables Required:
    LLAMAINDEX_API_KEY: Your LlamaIndex API key for document parsing

Optional Environment Variables:
    LLAMAINDEX_API_URL: API URL (default: https://api.llamaindex.ai)
    MAX_RETRIES: Maximum API retry attempts (default: 3)
    RETRY_DELAY: Base retry delay in seconds (default: 1)
    REQUEST_TIMEOUT: API request timeout in seconds (default: 30)
    MAX_FILE_SIZE_MB: Maximum file size in MB (default: 50)
    LOG_LEVEL: Logging level (default: INFO)
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from crew_feedback_parser.config.config_manager import ConfigManager, ConfigurationError


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="crew-feedback-parser",
        description="Process crew feedback forms and extract structured data to Excel",
        epilog="""
Examples:
  %(prog)s ./feedback_forms ./output/crew_feedback.xlsx
  %(prog)s ./feedback_forms ./output/crew_feedback.xlsx --verbose --workers 5
  %(prog)s ./feedback_forms ./output/crew_feedback.xlsx --single-file form1.pdf
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "input_folder",
        help="Path to folder containing feedback forms (PDF, PNG, JPG, TIFF)"
    )
    parser.add_argument(
        "output_file",
        help="Path to output Excel file (.xlsx extension recommended)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output except errors"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        metavar="N",
        help="Number of concurrent processing workers (default: 3, max: 10)"
    )
    
    parser.add_argument(
        "--single-file",
        metavar="FILENAME",
        help="Process only a specific file from the input folder"
    )
    
    parser.add_argument(
        "--report-file",
        metavar="PATH",
        help="Save detailed processing report to JSON file"
    )
    
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        help="Save logs to specified file (default: logs/crew_feedback_parser.log)"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and files, don't process"
    )
    
    parser.add_argument(
        "--skip-api-check",
        action="store_true",
        help="Skip API connection validation (useful for testing)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Crew Feedback Parser 1.0.0"
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments and provide helpful error messages.
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        SystemExit: If validation fails
    """
    # Validate input folder
    input_path = Path(args.input_folder)
    if not input_path.exists():
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        print("Please provide a valid path to a folder containing feedback forms")
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"Error: '{args.input_folder}' is not a directory")
        print("Please provide a path to a folder, not a file")
        sys.exit(1)
    
    # Validate output file path
    output_path = Path(args.output_file)
    
    # Check if output directory exists, create if needed
    output_dir = output_path.parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error: Cannot create output directory '{output_dir}': {e}")
            sys.exit(1)
    
    # Validate Excel file extension
    if not args.output_file.lower().endswith(('.xlsx', '.xls')):
        print("Warning: Output file should have .xlsx extension for best compatibility")
    
    # Validate worker count
    if args.workers < 1 or args.workers > 10:
        print("Error: Worker count must be between 1 and 10")
        sys.exit(1)
    
    # Validate single file if specified
    if args.single_file:
        single_file_path = input_path / args.single_file
        if not single_file_path.exists():
            print(f"Error: Specified file '{args.single_file}' not found in input folder")
            sys.exit(1)
        
        if not single_file_path.is_file():
            print(f"Error: '{args.single_file}' is not a file")
            sys.exit(1)
    
    # Validate conflicting options
    if args.verbose and args.quiet:
        print("Error: Cannot use both --verbose and --quiet options")
        sys.exit(1)


def print_usage_examples():
    """Print helpful usage examples."""
    print("\nUsage Examples:")
    print("  Basic processing:")
    print("    python main.py ./feedback_forms ./output/crew_feedback.xlsx")
    print()
    print("  With verbose logging:")
    print("    python main.py ./feedback_forms ./output/crew_feedback.xlsx --verbose")
    print()
    print("  Process single file:")
    print("    python main.py ./feedback_forms ./output/crew_feedback.xlsx --single-file form1.pdf")
    print()
    print("  With custom settings:")
    print("    python main.py ./feedback_forms ./output/crew_feedback.xlsx --workers 5 --report-file report.json")
    print()
    print("Environment Setup:")
    print("  Required: LLAMAINDEX_API_KEY=your_api_key_here")
    print("  Optional: LOG_LEVEL=DEBUG (for detailed logging)")
    print()


def validate_environment_setup(config_manager: ConfigManager, verbose: bool = False) -> bool:
    """
    Comprehensive validation of environment setup and configuration.
    
    Args:
        config_manager: Configuration manager instance
        verbose: Whether to print detailed validation information
        
    Returns:
        bool: True if all validations pass, False otherwise
    """
    validation_passed = True
    
    if verbose:
        print("\n" + "="*50)
        print("ENVIRONMENT VALIDATION")
        print("="*50)
    
    # 1. Validate API key
    try:
        if verbose:
            print("Checking LlamaIndex API key...")
        
        api_key = config_manager.get_api_key()
        if not config_manager.validate_api_key():
            print("✗ API key validation failed")
            print("  Issue: API key appears to be invalid or placeholder")
            print("  Solution: Set LLAMAINDEX_API_KEY environment variable with your actual API key")
            validation_passed = False
        else:
            if verbose:
                masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
                print(f"✓ API key validated: {masked_key}")
    
    except ConfigurationError as e:
        print(f"✗ API key configuration error: {e}")
        print("  Solution: Set LLAMAINDEX_API_KEY environment variable")
        validation_passed = False
    
    # 2. Validate API URL
    try:
        if verbose:
            print("Checking API URL configuration...")
        
        api_url = config_manager.get_api_url()
        if not api_url.startswith(('http://', 'https://')):
            print(f"✗ Invalid API URL format: {api_url}")
            print("  Solution: Ensure LLAMAINDEX_API_URL starts with http:// or https://")
            validation_passed = False
        else:
            if verbose:
                print(f"✓ API URL: {api_url}")
    
    except Exception as e:
        print(f"✗ API URL configuration error: {e}")
        validation_passed = False
    
    # 3. Validate numeric configurations
    try:
        if verbose:
            print("Checking numeric configuration values...")
        
        max_retries = config_manager.get_max_retries()
        retry_delay = config_manager.get_retry_delay()
        request_timeout = config_manager.get_request_timeout()
        max_file_size = config_manager.get_max_file_size_mb()
        
        # Validate ranges
        if max_retries < 0 or max_retries > 10:
            print(f"✗ Invalid MAX_RETRIES value: {max_retries} (should be 0-10)")
            validation_passed = False
        
        if retry_delay < 0 or retry_delay > 60:
            print(f"✗ Invalid RETRY_DELAY value: {retry_delay} (should be 0-60 seconds)")
            validation_passed = False
        
        if request_timeout < 5 or request_timeout > 300:
            print(f"✗ Invalid REQUEST_TIMEOUT value: {request_timeout} (should be 5-300 seconds)")
            validation_passed = False
        
        if max_file_size < 1 or max_file_size > 500:
            print(f"✗ Invalid MAX_FILE_SIZE_MB value: {max_file_size} (should be 1-500 MB)")
            validation_passed = False
        
        if verbose and validation_passed:
            print(f"✓ Max retries: {max_retries}")
            print(f"✓ Retry delay: {retry_delay}s")
            print(f"✓ Request timeout: {request_timeout}s")
            print(f"✓ Max file size: {max_file_size}MB")
    
    except Exception as e:
        print(f"✗ Numeric configuration error: {e}")
        validation_passed = False
    
    # 4. Validate supported formats
    try:
        if verbose:
            print("Checking supported file formats...")
        
        supported_formats = config_manager.get_supported_formats()
        expected_formats = {'pdf', 'png', 'jpg', 'jpeg', 'tiff'}
        
        if not supported_formats:
            print("✗ No supported formats configured")
            validation_passed = False
        else:
            # Check for common formats
            missing_formats = expected_formats - set(supported_formats)
            if missing_formats:
                print(f"⚠ Missing common formats: {', '.join(missing_formats)}")
            
            if verbose:
                print(f"✓ Supported formats: {', '.join(supported_formats)}")
    
    except Exception as e:
        print(f"✗ File format configuration error: {e}")
        validation_passed = False
    
    # 5. Validate log level
    try:
        if verbose:
            print("Checking logging configuration...")
        
        log_level = config_manager.get_log_level()
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        
        if log_level.upper() not in valid_levels:
            print(f"✗ Invalid LOG_LEVEL: {log_level} (should be one of: {', '.join(valid_levels)})")
            validation_passed = False
        else:
            if verbose:
                print(f"✓ Log level: {log_level}")
    
    except Exception as e:
        print(f"✗ Logging configuration error: {e}")
        validation_passed = False
    
    # 6. Check system dependencies
    if verbose:
        print("Checking system dependencies...")
    
    try:
        import requests
        if verbose:
            print("✓ requests library available")
    except ImportError:
        print("✗ Missing required library: requests")
        print("  Solution: pip install requests")
        validation_passed = False
    
    try:
        import openpyxl
        if verbose:
            print("✓ openpyxl library available")
    except ImportError:
        print("✗ Missing required library: openpyxl")
        print("  Solution: pip install openpyxl")
        validation_passed = False
    
    try:
        from dotenv import load_dotenv
        if verbose:
            print("✓ python-dotenv library available")
    except ImportError:
        print("✗ Missing required library: python-dotenv")
        print("  Solution: pip install python-dotenv")
        validation_passed = False
    
    # 7. Check .env file if it exists
    env_file = Path('.env')
    if env_file.exists():
        if verbose:
            print(f"✓ Found .env file: {env_file.absolute()}")
        
        # Check if .env file is readable
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                if 'LLAMAINDEX_API_KEY' in content:
                    if verbose:
                        print("✓ .env file contains LLAMAINDEX_API_KEY")
                else:
                    print("⚠ .env file exists but doesn't contain LLAMAINDEX_API_KEY")
        except Exception as e:
            print(f"⚠ Cannot read .env file: {e}")
    else:
        if verbose:
            print("ℹ No .env file found (using system environment variables)")
    
    if verbose:
        print("="*50)
        if validation_passed:
            print("✓ ALL VALIDATIONS PASSED")
        else:
            print("✗ VALIDATION FAILED - Please fix the issues above")
        print("="*50 + "\n")
    
    return validation_passed


def print_environment_help():
    """Print helpful information about environment setup."""
    print("\nEnvironment Setup Guide:")
    print("="*40)
    print()
    print("Required Environment Variables:")
    print("  LLAMAINDEX_API_KEY=your_actual_api_key_here")
    print()
    print("Optional Environment Variables:")
    print("  LLAMAINDEX_API_URL=https://api.llamaindex.ai")
    print("  MAX_RETRIES=3")
    print("  RETRY_DELAY=1")
    print("  REQUEST_TIMEOUT=30")
    print("  MAX_FILE_SIZE_MB=50")
    print("  SUPPORTED_FORMATS=pdf,png,jpg,jpeg,tiff")
    print("  LOG_LEVEL=INFO")
    print()
    print("Setup Methods:")
    print("  1. Create a .env file in the project root:")
    print("     echo 'LLAMAINDEX_API_KEY=your_key_here' > .env")
    print()
    print("  2. Set environment variables directly:")
    print("     export LLAMAINDEX_API_KEY=your_key_here")
    print()
    print("  3. Use system environment variables")
    print()
    print("To validate your setup, run:")
    print("  python main.py --validate-only input_folder output_file.xlsx")
    print()


def cleanup_resources(batch_processor=None, logging_config=None):
    """
    Clean up resources and perform final cleanup procedures.
    
    Args:
        batch_processor: BatchProcessor instance to clean up
        logging_config: LoggingConfig instance to clean up
    """
    try:
        if batch_processor:
            # Clear any error history to free memory
            batch_processor.clear_error_history()
        
        if logging_config:
            # Flush any remaining log messages
            import logging
            logging.shutdown()
    
    except Exception as e:
        # Don't let cleanup errors crash the application
        print(f"Warning: Cleanup error (non-critical): {e}")


def handle_processing_interruption(batch_processor=None, output_file=None):
    """
    Handle graceful interruption of processing workflow.
    
    Args:
        batch_processor: BatchProcessor instance
        output_file: Path to output file being written
    """
    print("\n⚠ Processing interrupted by user")
    
    if batch_processor and output_file:
        try:
            # Try to save any partial results
            results = batch_processor.get_processing_results()
            if results:
                print(f"Attempting to save {len(results)} partial results...")
                
                # Get processing statistics
                stats = batch_processor.get_processing_statistics()
                if stats['successful'] > 0:
                    print(f"✓ {stats['successful']} files were successfully processed")
                    print(f"Partial results may be available in: {output_file}")
                
        except Exception as e:
            print(f"Could not save partial results: {e}")
    
    print("Processing stopped gracefully")


def main():
    """Main application entry point with comprehensive CLI interface."""
    # Initialize variables for cleanup
    batch_processor = None
    logging_config = None
    
    # Create and parse arguments
    parser = create_argument_parser()
    
    # Handle no arguments case
    if len(sys.argv) == 1:
        parser.print_help()
        print_usage_examples()
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Determine log level
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "ERROR"
    else:
        log_level = "INFO"
    
    try:
        # Initialize configuration with comprehensive error handling
        if not args.quiet:
            print("Initializing Crew Feedback Parser...")
        
        config_manager = ConfigManager()
        
        if not args.quiet:
            print("✓ Configuration loaded successfully")
        
        # Comprehensive environment validation
        if not args.quiet:
            print("Validating environment setup...")
        
        validation_passed = validate_environment_setup(config_manager, verbose=args.verbose)
        
        if not validation_passed:
            print("\n✗ Environment validation failed!")
            print_environment_help()
            sys.exit(1)
        
        if not args.quiet:
            print("✓ Environment validation passed")
        
        # Handle validate-only mode
        if args.validate_only:
            print("\n✓ All validations completed successfully")
            print("Configuration is ready for processing")
            return
        
        # Print processing information
        if not args.quiet:
            print(f"\nProcessing Configuration:")
            print(f"  Input folder: {args.input_folder}")
            print(f"  Output file: {args.output_file}")
            print(f"  Workers: {args.workers}")
            if args.single_file:
                print(f"  Single file mode: {args.single_file}")
            if args.report_file:
                print(f"  Report file: {args.report_file}")
        
        # Initialize logging
        from crew_feedback_parser.utils.logging_config import setup_logging
        logging_config, error_handler = setup_logging(
            log_level=log_level,
            log_file=args.log_file,
            enable_console=not args.quiet
        )
        
        # Initialize batch processor
        from crew_feedback_parser.services.batch_processor import BatchProcessor
        batch_processor = BatchProcessor(config_manager)
        
        if not args.quiet:
            print("✓ Batch processor initialized")
        
        # Validate batch processor configuration
        if not batch_processor.validate_configuration(skip_api_check=args.skip_api_check):
            print("✗ Batch processor configuration validation failed")
            cleanup_resources(batch_processor, logging_config)
            sys.exit(1)
        
        if not args.quiet:
            print("✓ Batch processor configuration validated")
        
        # Execute processing workflow
        if not args.quiet:
            print(f"\nStarting processing workflow...")
            print("="*60)
        
        try:
            if args.single_file:
                # Process single file
                single_file_path = Path(args.input_folder) / args.single_file
                if not args.quiet:
                    print(f"Processing single file: {args.single_file}")
                
                result = batch_processor.process_single_file_standalone(
                    str(single_file_path),
                    args.output_file
                )
                
                # Print single file result
                if not args.quiet:
                    print(f"\nSingle file processing result:")
                    print(f"  File: {result.file_name}")
                    print(f"  Status: {result.status}")
                    if result.error_message:
                        print(f"  Error: {result.error_message}")
                    
                    if result.status == "pass":
                        print("✓ File processed successfully")
                    else:
                        print("✗ File processing failed")
            
            else:
                # Process entire directory
                if not args.quiet:
                    print(f"Processing directory: {args.input_folder}")
                    print(f"Using {args.workers} concurrent workers")
                
                processing_report = batch_processor.process_directory(
                    input_directory=args.input_folder,
                    output_excel_file=args.output_file,
                    max_workers=args.workers
                )
                
                # Print processing summary
                if not args.quiet:
                    batch_processor.print_processing_statistics()
                
                # Save detailed report if requested
                if args.report_file:
                    batch_processor.save_processing_report(args.report_file)
                    if not args.quiet:
                        print(f"✓ Detailed report saved to: {args.report_file}")
            
            if not args.quiet:
                print("="*60)
                print("✓ Processing completed successfully!")
                print(f"✓ Results saved to: {args.output_file}")
        
        except KeyboardInterrupt:
            handle_processing_interruption(batch_processor, args.output_file)
            cleanup_resources(batch_processor, logging_config)
            sys.exit(130)
        
        except Exception as processing_error:
            print(f"\n✗ Processing failed: {processing_error}")
            
            # Get error summary for debugging
            if batch_processor:
                error_summary = batch_processor.get_error_summary()
                if error_summary['total_errors'] > 0:
                    print(f"\nError Summary:")
                    print(f"  Total errors: {error_summary['total_errors']}")
                    if error_summary['most_common_error']:
                        print(f"  Most common: {error_summary['most_common_error']}")
            
            if args.verbose:
                import traceback
                traceback.print_exc()
            
            cleanup_resources(batch_processor, logging_config)
            sys.exit(1)
        
        # Successful completion - clean up resources
        cleanup_resources(batch_processor, logging_config)
        
    except ConfigurationError as e:
        print(f"\n✗ Configuration Error: {e}")
        print_environment_help()
        cleanup_resources(batch_processor, logging_config)
        sys.exit(1)
    except KeyboardInterrupt:
        handle_processing_interruption(batch_processor, getattr(args, 'output_file', None))
        cleanup_resources(batch_processor, logging_config)
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        print_environment_help()
        cleanup_resources(batch_processor, logging_config)
        sys.exit(1)


if __name__ == "__main__":
    main()