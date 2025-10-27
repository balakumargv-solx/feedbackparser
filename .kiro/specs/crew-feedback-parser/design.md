# Design Document

## Overview

The Crew Feedback Parser System is a Python-based batch processing application that leverages the LlamaIndex Cloud API to extract structured data from crew feedback forms. The system processes multiple documents in parallel, uses intelligent text extraction with the scientific preset, and outputs results to Excel format with comprehensive error handling and logging.

## Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Scanner  │───▶│  Document Parser │───▶│  Excel Writer   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Filter   │    │  Data Extractor  │    │  Report Gen.    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Error Handler  │    │   API Client     │    │    Logger       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components and Interfaces

### 1. Configuration Manager
- **Purpose**: Manages environment variables and system configuration
- **Key Methods**:
  - `load_config()`: Validates and loads API keys and settings
  - `get_api_key()`: Returns LlamaIndex API key
  - `get_output_settings()`: Returns Excel output configuration

### 2. File Scanner
- **Purpose**: Discovers and validates input files
- **Key Methods**:
  - `scan_directory(path)`: Returns list of valid document files
  - `filter_supported_formats()`: Filters for PDF, PNG, JPG, TIFF files
  - `validate_file_access()`: Checks file permissions and readability

### 3. LlamaIndex API Client
- **Purpose**: Handles communication with LlamaIndex Cloud API
- **Key Methods**:
  - `parse_document(file_path)`: Sends document to API for parsing
  - `handle_rate_limits()`: Implements exponential backoff retry logic
  - `validate_response()`: Checks API response validity

### 4. Data Extractor
- **Purpose**: Extracts structured data from parsed text using pattern matching and NLP
- **Key Methods**:
  - `extract_crew_info(text)`: Extracts vessel, crew name, and rank
  - `extract_ratings(text)`: Extracts 1-5 scale ratings for all metrics
  - `extract_feature_preference(text)`: Extracts preference text/selection
  - `validate_extracted_data()`: Ensures data quality and completeness

### 5. Excel Writer
- **Purpose**: Manages single Excel file with multiple sheets for data and tracking
- **Key Methods**:
  - `create_or_load_workbook()`: Creates new or loads existing Excel file with two sheets
  - `append_feedback_data()`: Adds new row to "Feedback_Data" sheet with extracted data
  - `update_tracking_sheet()`: Records processing status (pass/fail/error) in "Processing_Log" sheet
  - `format_columns()`: Applies proper formatting and data types to both sheets

### 6. Error Handler and Logger
- **Purpose**: Comprehensive error management and logging
- **Key Methods**:
  - `log_processing_status()`: Tracks file processing progress
  - `handle_api_errors()`: Manages API-specific error scenarios
  - `generate_error_report()`: Creates summary of failed files

## Data Models

### Excel File Structure
The system creates a single Excel file with two sheets:

**Sheet 1: "Feedback_Data"** - Contains extracted feedback data with exact columns:
- vessel (Text)
- crew_name (Text) 
- crew_rank (Text)
- safer_with_sos (Integer 1-5)
- fatigue_monitoring_prevention (Integer 1-5)
- geofence_awareness (Integer 1-5)
- heat_stress_alerts_change (Integer 1-5)
- work_rest_hour_monitoring (Integer 1-5)
- ptw_system_improvement (Integer 1-5)
- paperwork_error_reduction (Integer 1-5)
- noise_exposure_monitoring (Integer 1-5)
- activity_tracking_awareness (Integer 1-5)
- fall_detection_confidence (Integer 1-5)
- feature_preference (Text)

**Sheet 2: "Processing_Log"** - Tracks processing status with columns:
- file_name (Text)
- status (Text: "pass", "fail", "error")
- error_message (Text)
- processing_timestamp (DateTime)

### FeedbackData Class
```python
@dataclass
class FeedbackData:
    vessel: str
    crew_name: str
    crew_rank: str
    safer_with_sos: int  # 1-5
    fatigue_monitoring_prevention: int  # 1-5
    geofence_awareness: int  # 1-5
    heat_stress_alerts_change: int  # 1-5
    work_rest_hour_monitoring: int  # 1-5
    ptw_system_improvement: int  # 1-5
    paperwork_error_reduction: int  # 1-5
    noise_exposure_monitoring: int  # 1-5
    activity_tracking_awareness: int  # 1-5
    fall_detection_confidence: int  # 1-5
    feature_preference: str
```

### ProcessingResult Class
```python
@dataclass
class ProcessingResult:
    file_name: str
    status: str  # "pass", "fail", "error"
    data: Optional[FeedbackData]
    error_message: Optional[str]
    processing_timestamp: datetime
```

## Error Handling

### API Error Management
- **Rate Limiting**: Exponential backoff with jitter (1s, 2s, 4s, 8s delays)
- **Network Errors**: Retry up to 3 times with connection timeout handling
- **Authentication Errors**: Clear error messages with configuration guidance
- **Quota Exceeded**: Graceful degradation with partial processing reports

### File Processing Errors
- **Corrupted Files**: Skip and log with detailed error information
- **Unsupported Formats**: Filter out during scanning phase
- **Permission Issues**: Clear error messages with file path details
- **Large Files**: Size validation with appropriate error handling

### Data Extraction Errors
- **Missing Fields**: Partial data extraction with confidence scoring
- **Invalid Ratings**: Default values with manual review flags
- **Text Recognition Issues**: Quality indicators and manual review recommendations

## Testing Strategy

### Unit Testing
- **Configuration Manager**: Environment variable validation and error scenarios
- **Data Extractor**: Pattern matching accuracy with sample text inputs
- **Excel Writer**: File creation, data formatting, and append operations
- **API Client**: Mock API responses and error condition handling

### Integration Testing
- **End-to-End Processing**: Complete workflow with sample feedback forms
- **API Integration**: Real API calls with test documents
- **Excel Output Validation**: Verify data accuracy and formatting
- **Error Scenario Testing**: Simulate various failure conditions

### Performance Testing
- **Batch Processing**: Test with varying numbers of files (10, 50, 100+)
- **Large File Handling**: Test with high-resolution scanned documents
- **Concurrent Processing**: Validate thread safety and resource usage
- **Memory Usage**: Monitor memory consumption during large batch operations

## Security Considerations

- **API Key Management**: Environment variable storage with validation
- **File Access**: Proper permission checking and path validation
- **Data Privacy**: No sensitive data logging or temporary file retention
- **Input Validation**: Sanitize file paths and prevent directory traversal

## Performance Optimizations

- **Concurrent Processing**: Process multiple files simultaneously with thread pool
- **Caching**: Cache API responses for duplicate documents
- **Streaming**: Process large files without loading entirely into memory
- **Progress Tracking**: Real-time progress updates for long-running operations