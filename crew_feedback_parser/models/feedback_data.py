"""
Data models for crew feedback parsing system.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class FeedbackData:
    """
    Data model for crew feedback form data with exact field specifications
    matching the required Excel columns.
    """
    vessel: str
    crew_name: str
    crew_rank: str
    safer_with_sos: int  # 1-5 rating
    fatigue_monitoring_prevention: int  # 1-5 rating
    geofence_awareness: int  # 1-5 rating
    heat_stress_alerts_change: int  # 1-5 rating
    work_rest_hour_monitoring: int  # 1-5 rating
    ptw_system_improvement: int  # 1-5 rating
    paperwork_error_reduction: int  # 1-5 rating
    noise_exposure_monitoring: int  # 1-5 rating
    activity_tracking_awareness: int  # 1-5 rating
    fall_detection_confidence: int  # 1-5 rating
    feature_preference: str
    additional_comments: Optional[str] = None  # Additional feedback comments

    def validate_ratings(self) -> bool:
        """
        Validate that all rating fields are within the valid 1-5 range.
        
        Returns:
            bool: True if all ratings are valid, False otherwise
        """
        rating_fields = [
            self.safer_with_sos,
            self.fatigue_monitoring_prevention,
            self.geofence_awareness,
            self.heat_stress_alerts_change,
            self.work_rest_hour_monitoring,
            self.ptw_system_improvement,
            self.paperwork_error_reduction,
            self.noise_exposure_monitoring,
            self.activity_tracking_awareness,
            self.fall_detection_confidence
        ]
        
        return all(1 <= rating <= 5 for rating in rating_fields if rating is not None)

    def get_invalid_ratings(self) -> list[str]:
        """
        Get list of field names that have invalid ratings.
        
        Returns:
            list[str]: List of field names with invalid ratings
        """
        invalid_fields = []
        rating_field_map = {
            'safer_with_sos': self.safer_with_sos,
            'fatigue_monitoring_prevention': self.fatigue_monitoring_prevention,
            'geofence_awareness': self.geofence_awareness,
            'heat_stress_alerts_change': self.heat_stress_alerts_change,
            'work_rest_hour_monitoring': self.work_rest_hour_monitoring,
            'ptw_system_improvement': self.ptw_system_improvement,
            'paperwork_error_reduction': self.paperwork_error_reduction,
            'noise_exposure_monitoring': self.noise_exposure_monitoring,
            'activity_tracking_awareness': self.activity_tracking_awareness,
            'fall_detection_confidence': self.fall_detection_confidence
        }
        
        for field_name, rating in rating_field_map.items():
            if rating is not None and not (1 <= rating <= 5):
                invalid_fields.append(field_name)
        
        return invalid_fields

    def is_complete(self) -> bool:
        """
        Check if all required fields are populated.
        
        Returns:
            bool: True if all fields have values, False otherwise
        """
        return all([
            self.vessel,
            self.crew_name,
            self.crew_rank,
            self.safer_with_sos is not None,
            self.fatigue_monitoring_prevention is not None,
            self.geofence_awareness is not None,
            self.heat_stress_alerts_change is not None,
            self.work_rest_hour_monitoring is not None,
            self.ptw_system_improvement is not None,
            self.paperwork_error_reduction is not None,
            self.noise_exposure_monitoring is not None,
            self.activity_tracking_awareness is not None,
            self.fall_detection_confidence is not None,
            self.feature_preference
        ])


from datetime import datetime


@dataclass
class ProcessingResult:
    """
    Data model for tracking file processing status and results.
    """
    file_name: str
    status: str  # "pass", "fail", "error"
    error_message: Optional[str] = None
    processing_timestamp: Optional[datetime] = None
    data: Optional[FeedbackData] = None

    def __post_init__(self):
        """Set processing timestamp if not provided."""
        if self.processing_timestamp is None:
            self.processing_timestamp = datetime.now()

    def is_successful(self) -> bool:
        """
        Check if processing was successful.
        
        Returns:
            bool: True if status is "pass", False otherwise
        """
        return self.status == "pass"

    def has_data(self) -> bool:
        """
        Check if processing result contains extracted data.
        
        Returns:
            bool: True if data is present, False otherwise
        """
        return self.data is not None

    def get_error_summary(self) -> str:
        """
        Get a summary of the processing result for logging.
        
        Returns:
            str: Summary string with file name, status, and error if applicable
        """
        summary = f"File: {self.file_name}, Status: {self.status}"
        if self.error_message:
            summary += f", Error: {self.error_message}"
        return summary