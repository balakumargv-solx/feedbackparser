"""
Enhanced file validation utilities for crew feedback parser.

Provides additional validation logic for file filtering and error handling
as specified in requirement 4.1.
"""

import os
import mimetypes
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Represents a file validation error with details."""
    file_path: Path
    error_type: str
    error_message: str
    is_recoverable: bool = False


class FileValidator:
    """
    Enhanced file validation with detailed error handling and recovery options.
    
    Provides comprehensive validation beyond basic file access checking,
    including MIME type validation and file integrity checks.
    """
    
    # MIME types for supported formats
    SUPPORTED_MIME_TYPES = {
        'application/pdf',
        'image/png', 
        'image/jpeg',
        'image/tiff',
        'image/tif'
    }
    
    # Maximum file size (100MB) to prevent processing extremely large files
    MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_files_batch(self, file_paths: List[Path]) -> Tuple[List[Path], List[ValidationError]]:
        """
        Validate a batch of files and return valid files and errors.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Tuple of (valid_files, validation_errors)
        """
        valid_files = []
        validation_errors = []
        
        for file_path in file_paths:
            try:
                if self._validate_file_comprehensive(file_path):
                    valid_files.append(file_path)
                    self.logger.debug(f"File passed validation: {file_path}")
                else:
                    error = ValidationError(
                        file_path=file_path,
                        error_type="validation_failed",
                        error_message="File failed comprehensive validation",
                        is_recoverable=False
                    )
                    validation_errors.append(error)
                    
            except PermissionError as e:
                error = ValidationError(
                    file_path=file_path,
                    error_type="permission_error",
                    error_message=f"Permission denied: {str(e)}",
                    is_recoverable=False
                )
                validation_errors.append(error)
                self.logger.warning(f"Permission error for {file_path}: {e}")
                
            except OSError as e:
                error = ValidationError(
                    file_path=file_path,
                    error_type="os_error", 
                    error_message=f"OS error: {str(e)}",
                    is_recoverable=True  # Might be temporary
                )
                validation_errors.append(error)
                self.logger.warning(f"OS error for {file_path}: {e}")
                
            except Exception as e:
                error = ValidationError(
                    file_path=file_path,
                    error_type="unexpected_error",
                    error_message=f"Unexpected error: {str(e)}",
                    is_recoverable=False
                )
                validation_errors.append(error)
                self.logger.error(f"Unexpected error validating {file_path}: {e}")
        
        self.logger.info(f"Validation complete: {len(valid_files)} valid, {len(validation_errors)} errors")
        return valid_files, validation_errors
    
    def _validate_file_comprehensive(self, file_path: Path) -> bool:
        """
        Perform comprehensive file validation including MIME type and size checks.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            True if file passes all validation checks
            
        Raises:
            PermissionError: If file access is denied
            OSError: If file system error occurs
        """
        # Basic existence and access checks
        if not file_path.exists():
            self.logger.debug(f"File does not exist: {file_path}")
            return False
            
        if not file_path.is_file():
            self.logger.debug(f"Path is not a file: {file_path}")
            return False
            
        # Check file permissions
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"No read permission for file: {file_path}")
        
        # Check file size
        try:
            file_size = file_path.stat().st_size
            if file_size == 0:
                self.logger.debug(f"File is empty: {file_path}")
                return False
                
            if file_size > self.MAX_FILE_SIZE_BYTES:
                self.logger.warning(f"File too large ({file_size} bytes): {file_path}")
                return False
                
        except OSError as e:
            self.logger.error(f"Error getting file stats for {file_path}: {e}")
            raise
        
        # Validate MIME type
        if not self._validate_mime_type(file_path):
            return False
            
        # Test file readability
        if not self._test_file_readability(file_path):
            return False
            
        return True
    
    def _validate_mime_type(self, file_path: Path) -> bool:
        """
        Validate file MIME type matches supported formats.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if MIME type is supported
        """
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            if mime_type is None:
                # Try to determine from file extension
                extension = file_path.suffix.lower()
                if extension in ['.pdf']:
                    mime_type = 'application/pdf'
                elif extension in ['.png']:
                    mime_type = 'image/png'
                elif extension in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                elif extension in ['.tiff', '.tif']:
                    mime_type = 'image/tiff'
                else:
                    self.logger.debug(f"Unknown MIME type for {file_path}")
                    return False
            
            if mime_type not in self.SUPPORTED_MIME_TYPES:
                self.logger.debug(f"Unsupported MIME type {mime_type} for {file_path}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.warning(f"Error determining MIME type for {file_path}: {e}")
            return False
    
    def _test_file_readability(self, file_path: Path) -> bool:
        """
        Test if file can be opened and read.
        
        Args:
            file_path: Path to file to test
            
        Returns:
            True if file is readable
            
        Raises:
            PermissionError: If access is denied
            OSError: If file system error occurs
        """
        try:
            with open(file_path, 'rb') as f:
                # Try to read first 4KB to test readability
                f.read(4096)
            return True
            
        except PermissionError:
            raise  # Re-raise permission errors
            
        except (IOError, OSError) as e:
            self.logger.warning(f"File readability test failed for {file_path}: {e}")
            raise
            
        except Exception as e:
            self.logger.error(f"Unexpected error testing readability of {file_path}: {e}")
            return False
    
    def get_validation_summary(self, validation_errors: List[ValidationError]) -> Dict[str, int]:
        """
        Generate summary statistics for validation errors.
        
        Args:
            validation_errors: List of validation errors
            
        Returns:
            Dictionary with error type counts
        """
        summary = {}
        for error in validation_errors:
            error_type = error.error_type
            summary[error_type] = summary.get(error_type, 0) + 1
            
        return summary
    
    def filter_recoverable_errors(self, validation_errors: List[ValidationError]) -> List[ValidationError]:
        """
        Filter validation errors to return only recoverable ones.
        
        Args:
            validation_errors: List of all validation errors
            
        Returns:
            List of recoverable validation errors
        """
        return [error for error in validation_errors if error.is_recoverable]