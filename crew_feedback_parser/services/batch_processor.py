"""
Batch processing orchestrator for crew feedback parsing system.
Coordinates file scanning, API calls, and Excel writing with proper error isolation.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..config.config_manager import ConfigManager, ConfigurationError
from ..services.file_scanner import FileScanner
from ..services.llamaindex_client import LlamaIndexClient, LlamaIndexAPIError
from ..services.data_extractor import DataExtractor, ExtractionResult
from ..services.excel_writer import ExcelWriter
from ..models.feedback_data import ProcessingResult, FeedbackData
from ..utils.logging_config import ErrorHandler
from ..utils.report_generator import ReportGenerator


class BatchProcessingError(Exception):
    """Exception raised for batch processing errors."""
    pass


class BatchProcessor:
    """
    Orchestrates the complete batch processing workflow for crew feedback forms.
    
    Coordinates file scanning, document parsing via LlamaIndex API, data extraction,
    and Excel output generation with comprehensive error handling and logging.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize batch processor with configuration.
        
        Args:
            config_manager: Configuration manager instance
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize comprehensive error handling and reporting
        self.error_handler = ErrorHandler(self.logger)
        self.report_generator = ReportGenerator(self.logger)
        
        # Initialize service components
        self.file_scanner = FileScanner()
        
        # Choose parsing engine based on configuration
        import os
        parsing_engine = os.getenv('PARSING_ENGINE', 'llamaindex').lower()
        if parsing_engine == 'openai':
            from .openai_client import OpenAIClient
            self.document_client = OpenAIClient(config_manager)
            self.logger.info("Using OpenAI for document parsing")
        else:
            from .llamaindex_client import LlamaIndexClient
            self.document_client = LlamaIndexClient(config_manager)
            self.logger.info("Using LlamaIndex for document parsing")
            
        self.data_extractor = DataExtractor(config_manager)
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None,
            'processing_duration': 0.0
        }
        
        # Processing results storage
        self.processing_results: List[ProcessingResult] = []
        
        self.logger.info("Batch processor initialized successfully with comprehensive error handling")

    def process_directory(self, 
                         input_directory: str, 
                         output_excel_file: str,
                         max_workers: int = 3) -> Dict[str, Any]:
        """
        Process all feedback forms in a directory and generate Excel output.
        
        Args:
            input_directory: Path to directory containing feedback form files
            output_excel_file: Path to output Excel file
            max_workers: Maximum number of concurrent processing threads
            
        Returns:
            Dict containing processing summary and statistics
            
        Raises:
            BatchProcessingError: If processing fails
            FileNotFoundError: If input directory doesn't exist
        """
        self.logger.info(f"Starting batch processing - Input: {input_directory}, Output: {output_excel_file}")
        
        # Reset statistics
        self._reset_stats()
        self.stats['start_time'] = datetime.now()
        
        try:
            # Step 1: Scan directory for valid files
            self.logger.info("Step 1: Scanning directory for feedback form files")
            valid_files = self._scan_and_validate_files(input_directory)
            
            if not valid_files:
                raise BatchProcessingError(f"No valid feedback form files found in {input_directory}")
            
            self.stats['total_files'] = len(valid_files)
            self.logger.info(f"Found {len(valid_files)} valid files to process")
            
            # Step 2: Initialize Excel writer
            self.logger.info("Step 2: Initializing Excel output file")
            excel_writer = self._initialize_excel_writer(output_excel_file)
            
            # Step 3: Process files with error isolation
            self.logger.info(f"Step 3: Processing files with {max_workers} concurrent workers")
            self._process_files_concurrently(valid_files, excel_writer, max_workers)
            
            # Step 4: Save Excel file and generate summary
            self.logger.info("Step 4: Saving Excel file and generating summary")
            excel_writer.save_workbook()
            excel_writer.close_workbook()
            
            # Calculate final statistics
            self.stats['end_time'] = datetime.now()
            self.stats['processing_duration'] = (
                self.stats['end_time'] - self.stats['start_time']
            ).total_seconds()
            
            # Generate comprehensive processing report
            error_summary = self.error_handler.get_error_summary()
            comprehensive_report = self.report_generator.generate_processing_report(
                processing_results=self.processing_results,
                processing_stats=self.stats,
                error_summary=error_summary
            )
            
            # Log comprehensive error summary
            self.error_handler.log_error_summary()
            
            # Log summary text report
            summary_text = self.report_generator.generate_summary_text(comprehensive_report)
            self.logger.info("Processing Summary Report:")
            for line in summary_text.split('\n'):
                if line.strip():
                    self.logger.info(line)
            
            self.logger.info(f"Batch processing completed successfully in {self.stats['processing_duration']:.1f}s")
            
            return comprehensive_report
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            self.stats['end_time'] = datetime.now()
            if self.stats['start_time']:
                self.stats['processing_duration'] = (
                    self.stats['end_time'] - self.stats['start_time']
                ).total_seconds()
            
            # Log error summary even on failure
            self.error_handler.log_error_summary()
            
            raise BatchProcessingError(f"Batch processing failed: {e}") from e

    def _scan_and_validate_files(self, input_directory: str) -> List[Path]:
        """
        Scan directory and validate files for processing.
        
        Args:
            input_directory: Directory path to scan
            
        Returns:
            List of valid file paths
            
        Raises:
            BatchProcessingError: If scanning fails
        """
        try:
            # Use enhanced validation for better error reporting
            valid_files, validation_errors = self.file_scanner.scan_directory_with_enhanced_validation(
                input_directory
            )
            
            # Handle validation errors gracefully
            if validation_errors:
                self.file_scanner.handle_validation_errors_gracefully(validation_errors)
                self.logger.warning(f"File validation found {len(validation_errors)} issues, "
                                  f"but {len(valid_files)} files are still processable")
            
            return valid_files
            
        except Exception as e:
            raise BatchProcessingError(f"Failed to scan directory {input_directory}: {e}") from e

    def _initialize_excel_writer(self, output_excel_file: str) -> ExcelWriter:
        """
        Initialize Excel writer with proper error handling.
        
        Args:
            output_excel_file: Path to output Excel file
            
        Returns:
            Initialized ExcelWriter instance
            
        Raises:
            BatchProcessingError: If Excel initialization fails
        """
        try:
            excel_writer = ExcelWriter(output_excel_file)
            excel_writer.create_or_load_workbook()
            
            # Apply formatting to existing data if loading existing file
            if Path(output_excel_file).exists():
                excel_writer.apply_formatting_to_existing_data()
                self.logger.info(f"Loaded existing Excel file: {output_excel_file}")
            else:
                self.logger.info(f"Created new Excel file: {output_excel_file}")
            
            return excel_writer
            
        except Exception as e:
            raise BatchProcessingError(f"Failed to initialize Excel file {output_excel_file}: {e}") from e

    def _process_files_concurrently(self, 
                                  files: List[Path], 
                                  excel_writer: ExcelWriter,
                                  max_workers: int) -> None:
        """
        Process files concurrently with proper error isolation.
        
        Args:
            files: List of file paths to process
            excel_writer: Excel writer instance
            max_workers: Maximum number of concurrent workers
        """
        # Process files in batches to avoid overwhelming the API
        batch_size = min(max_workers * 2, len(files))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in files
            }
            
            # Process completed futures as they finish
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    processing_result = future.result()
                    self.processing_results.append(processing_result)
                    
                    # Update statistics
                    if processing_result.status == "pass":
                        self.stats['successful'] += 1
                    elif processing_result.status == "fail":
                        self.stats['failed'] += 1
                    else:  # error
                        self.stats['errors'] += 1
                    
                    # Write to Excel immediately to avoid memory issues
                    excel_writer.append_feedback_and_log(processing_result)
                    
                    # Log progress
                    processed_count = self.stats['successful'] + self.stats['failed'] + self.stats['errors']
                    self.logger.info(f"Processed {processed_count}/{self.stats['total_files']}: "
                                   f"{file_path.name} - {processing_result.status}")
                    
                except Exception as e:
                    # Handle unexpected errors in future execution with comprehensive error handling
                    error_details = self.error_handler.handle_file_error(
                        str(file_path), e, "concurrent processing"
                    )
                    
                    error_result = ProcessingResult(
                        file_name=file_path.name,
                        status="error",
                        error_message=f"Unexpected processing error: {e}"
                    )
                    self.processing_results.append(error_result)
                    self.stats['errors'] += 1
                    
                    # Still write error to Excel
                    excel_writer.append_feedback_and_log(error_result)

    def _process_single_file(self, file_path: Path) -> ProcessingResult:
        """
        Process a single feedback form file with comprehensive error handling.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            ProcessingResult with processing status and data
        """
        file_name = file_path.name
        self.logger.debug(f"Processing file: {file_name}")
        
        try:
            # Step 1: Parse document using LlamaIndex API
            self.logger.debug(f"Parsing document: {file_name}")
            parse_result = self.document_client.parse_document(str(file_path))
            
            if not parse_result.get('text'):
                return ProcessingResult(
                    file_name=file_name,
                    status="fail",
                    error_message="No text extracted from document"
                )
            
            # Step 2: Extract structured data
            self.logger.debug(f"Extracting data from: {file_name}")
            
            # Use structured data directly if available from OpenAI JSON response
            structured_data = parse_result.get('structured_data')
            if structured_data:
                self.logger.info(f"Using OpenAI structured JSON data for {file_name}")
                extraction_result = self._create_extraction_result_from_json(structured_data, file_name)
            else:
                # Fall back to pattern matching
                extraction_result = self.data_extractor.extract_complete_data(parse_result['text'], file_name)
            
            if not extraction_result.data:
                return ProcessingResult(
                    file_name=file_name,
                    status="fail",
                    error_message="Failed to extract structured data from text"
                )
            
            # Step 3: Validate extraction quality
            if self.data_extractor.flag_for_manual_review(extraction_result):
                status = "fail"
                error_message = (f"Low quality extraction (confidence: {extraction_result.confidence_score:.2f}, "
                               f"missing: {len(extraction_result.missing_fields)} fields) - manual review required")
                
                # Log detailed extraction issues for debugging
                self.logger.warning(f"Quality issues for {file_name}: "
                                  f"confidence={extraction_result.confidence_score:.2f}, "
                                  f"missing_fields={extraction_result.missing_fields}, "
                                  f"notes={extraction_result.extraction_notes}")
            else:
                status = "pass"
                error_message = None
                
                # Log successful extraction details
                self.logger.debug(f"Successful extraction for {file_name}: "
                                f"confidence={extraction_result.confidence_score:.2f}, "
                                f"missing_fields={len(extraction_result.missing_fields)}")
            
            return ProcessingResult(
                file_name=file_name,
                status=status,
                error_message=error_message,
                data=extraction_result.data
            )
            
        except LlamaIndexAPIError as e:
            # Handle API errors with comprehensive error handling
            error_details = self.error_handler.handle_api_error(e, file_name)
            
            return ProcessingResult(
                file_name=file_name,
                status="error",
                error_message=f"API error: {e}"
            )
            
        except Exception as e:
            # Handle extraction and other errors
            if "extract" in str(e).lower() or "pattern" in str(e).lower():
                text_length = len(parse_result.get('text', '')) if 'parse_result' in locals() else 0
                error_details = self.error_handler.handle_extraction_error(e, file_name, text_length)
            else:
                error_details = self.error_handler.handle_file_error(str(file_path), e, "processing")
            
            return ProcessingResult(
                file_name=file_name,
                status="error",
                error_message=f"Processing error: {e}"
            )

    def _create_extraction_result_from_json(self, structured_data: Dict[str, Any], file_name: str) -> 'ExtractionResult':
        """
        Create ExtractionResult from OpenAI structured JSON data.
        
        Args:
            structured_data: JSON data from OpenAI
            file_name: Source filename
            
        Returns:
            ExtractionResult with high confidence
        """
        from ..models.feedback_data import FeedbackData
        from ..services.data_extractor import ExtractionResult
        
        # Create FeedbackData from structured JSON
        feedback_data = FeedbackData(
            vessel=structured_data.get('vessel'),
            crew_name=structured_data.get('crew_name'),
            crew_rank=structured_data.get('crew_rank'),
            safer_with_sos=structured_data.get('safer_with_sos'),
            fatigue_monitoring_prevention=structured_data.get('fatigue_monitoring_prevention'),
            geofence_awareness=structured_data.get('geofence_awareness'),
            heat_stress_alerts_change=structured_data.get('heat_stress_alerts_change'),
            work_rest_hour_monitoring=structured_data.get('work_rest_hour_monitoring'),
            ptw_system_improvement=structured_data.get('ptw_system_improvement'),
            paperwork_error_reduction=structured_data.get('paperwork_error_reduction'),
            noise_exposure_monitoring=structured_data.get('noise_exposure_monitoring'),
            activity_tracking_awareness=structured_data.get('activity_tracking_awareness'),
            fall_detection_confidence=structured_data.get('fall_detection_confidence'),
            feature_preference=structured_data.get('feature_preference')
        )
        
        # Count missing fields
        missing_fields = []
        for field_name, field_value in structured_data.items():
            if field_value is None and field_name != 'additional_comments':
                missing_fields.append(field_name)
        
        # Calculate confidence (high for OpenAI JSON, reduced by missing fields)
        base_confidence = 0.95
        confidence_penalty = len(missing_fields) * 0.05
        confidence_score = max(0.5, base_confidence - confidence_penalty)
        
        self.logger.info(f"OpenAI JSON extraction - Confidence: {confidence_score:.2f}, Missing: {len(missing_fields)} fields")
        
        return ExtractionResult(
            data=feedback_data,
            confidence_score=confidence_score,
            missing_fields=missing_fields,
            extraction_notes=[f"Extracted using OpenAI structured JSON for {file_name}"]
        )

    def _reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None,
            'processing_duration': 0.0
        }
        self.processing_results.clear()

    def generate_detailed_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate detailed processing report with comprehensive analysis.
        
        Args:
            output_file: Optional path to save report as JSON file
            
        Returns:
            Dict containing comprehensive processing report
        """
        error_summary = self.error_handler.get_error_summary()
        return self.report_generator.generate_processing_report(
            processing_results=self.processing_results,
            processing_stats=self.stats,
            error_summary=error_summary,
            output_file=output_file
        )

    def get_processing_summary_text(self) -> str:
        """
        Get human-readable processing summary text.
        
        Returns:
            Formatted text summary of processing results
        """
        if not self.processing_results:
            return "No processing results available"
        
        report = self.generate_detailed_report()
        return self.report_generator.generate_summary_text(report)

    def save_processing_report(self, output_file: str) -> None:
        """
        Save comprehensive processing report to file.
        
        Args:
            output_file: Path to save the report JSON file
        """
        try:
            report = self.generate_detailed_report(output_file)
            self.logger.info(f"Processing report saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save processing report: {e}")
            raise BatchProcessingError(f"Failed to save processing report: {e}") from e

    def get_processing_results(self) -> List[ProcessingResult]:
        """
        Get all processing results from the last batch operation.
        
        Returns:
            List of ProcessingResult objects
        """
        return self.processing_results.copy()

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dict containing current statistics
        """
        return self.stats.copy()

    def validate_configuration(self, skip_api_check: bool = False) -> bool:
        """
        Validate that all required configuration is properly set.
        
        Args:
            skip_api_check: If True, skip API connection validation
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Test API connection (unless skipped)
            if not skip_api_check:
                if not self.llamaindex_client.validate_connection():
                    self.logger.error("LlamaIndex API connection validation failed")
                    return False
            else:
                self.logger.info("Skipping API connection validation as requested")
            
            # Validate API key
            if not self.config.validate_api_key():
                self.logger.error("Invalid API key configuration")
                return False
            
            self.logger.info("Configuration validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def process_single_file_standalone(self, 
                                     file_path: str, 
                                     output_excel_file: str) -> ProcessingResult:
        """
        Process a single file independently (useful for testing or individual processing).
        
        Args:
            file_path: Path to single file to process
            output_excel_file: Path to output Excel file
            
        Returns:
            ProcessingResult with processing status and data
            
        Raises:
            BatchProcessingError: If processing fails
        """
        self.logger.info(f"Processing single file: {file_path}")
        
        try:
            # Validate file
            file_path_obj = Path(file_path)
            validation_result = self.file_scanner.validate_single_file(str(file_path_obj))
            
            if not validation_result.is_valid:
                raise BatchProcessingError(f"File validation failed: {validation_result.error_message}")
            
            # Initialize Excel writer
            excel_writer = self._initialize_excel_writer(output_excel_file)
            
            # Process the file
            processing_result = self._process_single_file(file_path_obj)
            
            # Write to Excel
            excel_writer.append_feedback_and_log(processing_result)
            excel_writer.save_workbook()
            excel_writer.close_workbook()
            
            # Log error summary for single file processing
            self.error_handler.log_error_summary()
            
            self.logger.info(f"Single file processing completed: {processing_result.status}")
            return processing_result
            
        except Exception as e:
            # Handle single file processing errors
            error_details = self.error_handler.handle_file_error(file_path, e, "standalone processing")
            raise BatchProcessingError(f"Single file processing failed: {e}") from e

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive error summary from the error handler.
        
        Returns:
            Dict containing detailed error information and statistics
        """
        return self.error_handler.get_error_summary()

    def clear_error_history(self) -> None:
        """Clear error history and statistics for a fresh start."""
        self.error_handler.clear_error_history()
        self.logger.info("Error history cleared for batch processor")

    def handle_partial_failures(self, continue_on_errors: bool = True) -> None:
        """
        Configure how to handle partial failures during batch processing.
        
        Args:
            continue_on_errors: Whether to continue processing remaining files when errors occur
        """
        self.continue_on_errors = continue_on_errors
        self.logger.info(f"Partial failure handling configured: continue_on_errors={continue_on_errors}")

    def log_processing_progress(self, processed_count: int, total_count: int, current_file: str) -> None:
        """
        Log processing progress with detailed information.
        
        Args:
            processed_count: Number of files processed so far
            total_count: Total number of files to process
            current_file: Name of current file being processed
        """
        progress_percent = (processed_count / total_count * 100) if total_count > 0 else 0
        
        self.logger.info(f"Progress: {processed_count}/{total_count} ({progress_percent:.1f}%) - "
                        f"Processing: {current_file}")
        
        # Log intermediate statistics every 10 files
        if processed_count % 10 == 0 and processed_count > 0:
            self.logger.info(f"Intermediate stats - Success: {self.stats['successful']}, "
                           f"Failed: {self.stats['failed']}, Errors: {self.stats['errors']}")
            
            # Log error summary if there are errors
            if self.stats['errors'] > 0 or self.stats['failed'] > 0:
                error_summary = self.error_handler.get_error_summary()
                if error_summary['total_errors'] > 0:
                    self.logger.warning(f"Error types encountered: {list(error_summary['error_counts_by_type'].keys())}")

    def get_success_failure_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics on successful vs failed processing.
        
        Returns:
            Dict containing success/failure statistics and analysis
        """
        total_processed = len(self.processing_results)
        
        if total_processed == 0:
            return {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'errors': 0,
                'success_rate': 0.0,
                'failure_rate': 0.0,
                'error_rate': 0.0,
                'analysis': 'No files processed'
            }
        
        successful = self.stats['successful']
        failed = self.stats['failed']
        errors = self.stats['errors']
        
        success_rate = (successful / total_processed) * 100
        failure_rate = (failed / total_processed) * 100
        error_rate = (errors / total_processed) * 100
        
        # Analyze success patterns
        analysis = []
        if success_rate >= 90:
            analysis.append("Excellent success rate (≥90%)")
        elif success_rate >= 70:
            analysis.append("Good success rate (70-89%)")
        elif success_rate >= 50:
            analysis.append("Moderate success rate (50-69%)")
        else:
            analysis.append("Low success rate (<50%) - requires attention")
        
        if error_rate > 20:
            analysis.append("High error rate (>20%) - check system configuration")
        elif error_rate > 10:
            analysis.append("Moderate error rate (10-20%) - monitor system health")
        
        if failure_rate > 30:
            analysis.append("High failure rate (>30%) - review document quality and extraction patterns")
        
        return {
            'total_files': total_processed,
            'successful': successful,
            'failed': failed,
            'errors': errors,
            'success_rate_percent': round(success_rate, 2),
            'failure_rate_percent': round(failure_rate, 2),
            'error_rate_percent': round(error_rate, 2),
            'analysis': analysis,
            'processing_duration_seconds': self.stats.get('processing_duration', 0),
            'files_per_second': round(total_processed / self.stats['processing_duration'], 2) if self.stats.get('processing_duration', 0) > 0 else 0
        }

    def print_processing_statistics(self) -> None:
        """Print formatted processing statistics to console and log."""
        stats = self.get_success_failure_statistics()
        
        print("\n" + "="*60)
        print("CREW FEEDBACK PROCESSING STATISTICS")
        print("="*60)
        print(f"Total files processed: {stats['total_files']}")
        print(f"Successful extractions: {stats['successful']} ({stats['success_rate_percent']:.1f}%)")
        print(f"Failed extractions: {stats['failed']} ({stats['failure_rate_percent']:.1f}%)")
        print(f"Processing errors: {stats['errors']} ({stats['error_rate_percent']:.1f}%)")
        print(f"Processing time: {stats['processing_duration_seconds']:.1f} seconds")
        print(f"Throughput: {stats['files_per_second']:.2f} files/second")
        
        if stats['analysis']:
            print("\nAnalysis:")
            for analysis_point in stats['analysis']:
                print(f"  • {analysis_point}")
        
        print("="*60 + "\n")
        
        # Also log the statistics
        self.logger.info("Processing statistics summary:")
        self.logger.info(f"  Total: {stats['total_files']}, Success: {stats['successful']}, "
                        f"Failed: {stats['failed']}, Errors: {stats['errors']}")
        self.logger.info(f"  Success rate: {stats['success_rate_percent']:.1f}%, "
                        f"Throughput: {stats['files_per_second']:.2f} files/sec")