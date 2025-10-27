"""
File scanner module for discovering and validating crew feedback form files.

This module provides functionality to scan directories for supported file formats
(PDF, PNG, JPG, TIFF) and validate file access permissions.
"""

import os
import logging
from pathlib import Path
from typing import List, Set, Tuple
from dataclasses import dataclass

from ..utils.file_validator import FileValidator, ValidationError


@dataclass
class FileValidationResult:
    """Result of file validation containing file path and validation status."""
    file_path: Path
    is_valid: bool
    error_message: str = ""


class FileScanner:
    """
    Handles scanning directories for supported feedback form files and validation.
    
    Supports PDF and common image formats (PNG, JPG, JPEG, TIFF) as specified
    in requirements 1.1 and 4.1.
    """
    
    # Supported file extensions for crew feedback forms
    SUPPORTED_EXTENSIONS: Set[str] = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_validator = FileValidator()
    
    def scan_directory(self, directory_path: str) -> List[Path]:
        """
        Scan directory for supported file formats.
        
        Args:
            directory_path: Path to directory containing feedback form files
            
        Returns:
            List of Path objects for valid, accessible files
            
        Raises:
            FileNotFoundError: If directory doesn't exist
            PermissionError: If directory is not accessible
        """
        try:
            dir_path = Path(directory_path)
            
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            if not dir_path.is_dir():
                raise ValueError(f"Path is not a directory: {directory_path}")
            
            # Check directory permissions
            if not os.access(dir_path, os.R_OK):
                raise PermissionError(f"No read permission for directory: {directory_path}")
            
            self.logger.info(f"Scanning directory: {directory_path}")
            
            # Get all files in directory
            all_files = []
            try:
                all_files = [f for f in dir_path.iterdir() if f.is_file()]
            except PermissionError as e:
                self.logger.error(f"Permission denied accessing directory contents: {e}")
                raise
            
            # Filter for supported formats
            supported_files = self._filter_supported_formats(all_files)
            
            # Validate file access
            valid_files = []
            for file_path in supported_files:
                validation_result = self._validate_file_access(file_path)
                if validation_result.is_valid:
                    valid_files.append(file_path)
                else:
                    self.logger.warning(f"File validation failed for {file_path}: {validation_result.error_message}")
            
            self.logger.info(f"Found {len(valid_files)} valid files out of {len(all_files)} total files")
            return valid_files
            
        except Exception as e:
            self.logger.error(f"Error scanning directory {directory_path}: {e}")
            raise
    
    def _filter_supported_formats(self, files: List[Path]) -> List[Path]:
        """
        Filter files by supported formats.
        
        Args:
            files: List of file paths to filter
            
        Returns:
            List of files with supported extensions
        """
        supported_files = []
        
        for file_path in files:
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                supported_files.append(file_path)
                self.logger.debug(f"Found supported file: {file_path}")
            else:
                self.logger.debug(f"Skipping unsupported file: {file_path}")
        
        return supported_files
    
    def _validate_file_access(self, file_path: Path) -> FileValidationResult:
        """
        Validate file access permissions and readability.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            FileValidationResult with validation status and error message
        """
        try:
            # Check if file exists
            if not file_path.exists():
                return FileValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    error_message="File does not exist"
                )
            
            # Check if it's actually a file
            if not file_path.is_file():
                return FileValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    error_message="Path is not a file"
                )
            
            # Check read permissions
            if not os.access(file_path, os.R_OK):
                return FileValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    error_message="No read permission for file"
                )
            
            # Check file size (avoid empty files)
            if file_path.stat().st_size == 0:
                return FileValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    error_message="File is empty"
                )
            
            # Try to open file to ensure it's readable
            try:
                with open(file_path, 'rb') as f:
                    # Read first few bytes to ensure file is accessible
                    f.read(1024)
            except (IOError, OSError) as e:
                return FileValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    error_message=f"File read error: {str(e)}"
                )
            
            return FileValidationResult(
                file_path=file_path,
                is_valid=True,
                error_message=""
            )
            
        except Exception as e:
            return FileValidationResult(
                file_path=file_path,
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    def get_supported_extensions(self) -> Set[str]:
        """
        Get the set of supported file extensions.
        
        Returns:
            Set of supported file extensions
        """
        return self.SUPPORTED_EXTENSIONS.copy()
    
    def validate_single_file(self, file_path: str) -> FileValidationResult:
        """
        Validate a single file for processing.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            FileValidationResult with validation status
        """
        path_obj = Path(file_path)
        
        # Check if file has supported extension
        if path_obj.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return FileValidationResult(
                file_path=path_obj,
                is_valid=False,
                error_message=f"Unsupported file format: {path_obj.suffix}"
            )
        
        return self._validate_file_access(path_obj)
    
    def scan_directory_with_enhanced_validation(self, directory_path: str) -> Tuple[List[Path], List[ValidationError]]:
        """
        Scan directory with enhanced validation and detailed error reporting.
        
        Args:
            directory_path: Path to directory containing feedback form files
            
        Returns:
            Tuple of (valid_files, validation_errors)
            
        Raises:
            FileNotFoundError: If directory doesn't exist
            PermissionError: If directory is not accessible
        """
        try:
            dir_path = Path(directory_path)
            
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            if not dir_path.is_dir():
                raise ValueError(f"Path is not a directory: {directory_path}")
            
            # Check directory permissions
            if not os.access(dir_path, os.R_OK):
                raise PermissionError(f"No read permission for directory: {directory_path}")
            
            self.logger.info(f"Scanning directory with enhanced validation: {directory_path}")
            
            # Get all files in directory
            all_files = []
            try:
                all_files = [f for f in dir_path.iterdir() if f.is_file()]
            except PermissionError as e:
                self.logger.error(f"Permission denied accessing directory contents: {e}")
                raise
            
            # Filter for supported formats first
            supported_files = self._filter_supported_formats(all_files)
            
            # Use enhanced validation
            valid_files, validation_errors = self.file_validator.validate_files_batch(supported_files)
            
            # Log summary
            error_summary = self.file_validator.get_validation_summary(validation_errors)
            self.logger.info(f"Enhanced validation complete: {len(valid_files)} valid files, "
                           f"{len(validation_errors)} errors. Error summary: {error_summary}")
            
            return valid_files, validation_errors
            
        except Exception as e:
            self.logger.error(f"Error in enhanced directory scan {directory_path}: {e}")
            raise
    
    def handle_validation_errors_gracefully(self, validation_errors: List[ValidationError]) -> None:
        """
        Handle validation errors gracefully with appropriate logging and recovery suggestions.
        
        Args:
            validation_errors: List of validation errors to handle
        """
        if not validation_errors:
            return
            
        # Group errors by type for better reporting
        error_summary = self.file_validator.get_validation_summary(validation_errors)
        
        self.logger.warning(f"File validation encountered {len(validation_errors)} errors:")
        for error_type, count in error_summary.items():
            self.logger.warning(f"  {error_type}: {count} files")
        
        # Log individual errors with suggestions
        for error in validation_errors:
            if error.error_type == "permission_error":
                self.logger.error(f"Permission denied for {error.file_path}. "
                                f"Check file permissions and try again.")
            elif error.error_type == "os_error" and error.is_recoverable:
                self.logger.warning(f"Temporary error for {error.file_path}: {error.error_message}. "
                                  f"File may be processed in retry attempt.")
            else:
                self.logger.error(f"Validation failed for {error.file_path}: {error.error_message}")
        
        # Suggest recovery actions for recoverable errors
        recoverable_errors = self.file_validator.filter_recoverable_errors(validation_errors)
        if recoverable_errors:
            self.logger.info(f"{len(recoverable_errors)} errors may be recoverable. "
                           f"Consider retrying these files later.")