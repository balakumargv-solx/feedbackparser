"""
OpenAI API client for document parsing and text extraction.
Uses GPT-4 Vision to directly analyze PDF documents and extract structured data.
"""
import os
import base64
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import requests
import json
from PIL import Image
import fitz  # PyMuPDF for PDF to image conversion

from ..config.config_manager import ConfigManager, ConfigurationError


class OpenAIAPIError(Exception):
    """Exception raised for OpenAI API-related errors."""
    pass


class OpenAIClient:
    """
    Client for interacting with OpenAI API for document parsing.
    Uses GPT-4 Vision to analyze documents and extract structured crew feedback data.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize OpenAI API client.
        
        Args:
            config_manager: Configuration manager instance
            
        Raises:
            ConfigurationError: If API configuration is invalid
        """
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Get OpenAI API key from environment
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ConfigurationError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # API configuration
        self.base_url = "https://api.openai.com/v1"
        self.model = "gpt-4o"  # Latest GPT-4 with enhanced vision capabilities
        self.max_tokens = 4000
        self.temperature = 0.05  # Very low temperature for consistent extraction of handwritten text
        
        self.logger.info("OpenAI API client initialized successfully")

    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a document using OpenAI GPT-4 Vision.
        
        Args:
            file_path: Path to the document file to parse
            
        Returns:
            Dict containing parsed text and metadata
            
        Raises:
            OpenAIAPIError: If API request fails
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.info(f"Parsing document with OpenAI: {file_path.name}")
        
        try:
            # Convert PDF to images if needed
            if file_path.suffix.lower() == '.pdf':
                images = self._pdf_to_images(file_path)
            else:
                # For image files, use directly
                images = [file_path]
            
            # Process each page/image
            all_data = []
            for i, image_path in enumerate(images):
                self.logger.debug(f"Processing page {i+1}/{len(images)}")
                
                # Encode image to base64
                image_base64 = self._encode_image_to_base64(image_path)
                
                # Call OpenAI API to get JSON data
                json_response = self._call_openai_vision_api(image_base64, file_path.name, i+1)
                
                try:
                    # Parse JSON response
                    page_data = json.loads(json_response)
                    all_data.append(page_data)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON from page {i+1}: {e}")
                    # Fallback to text format
                    all_data.append({"raw_text": json_response})
            
            # Combine data from all pages - handle multiple feedbacks intelligently
            combined_data = self._combine_page_data(all_data, file_path.name)
            
            # Convert structured data back to text format for compatibility
            combined_text = self._convert_json_to_text(combined_data)
            
            # Clean up temporary image files if we created them
            if file_path.suffix.lower() == '.pdf':
                for img_path in images:
                    if img_path != file_path:  # Don't delete the original PDF
                        try:
                            os.unlink(img_path)
                        except:
                            pass
            
            result = {
                'text': combined_text,
                'filename': file_path.name,
                'pages': len(images),
                'confidence': 0.95,  # OpenAI typically has high confidence
                'processing_time': 0.0,
                'metadata': {
                    'model': self.model,
                    'api': 'openai'
                },
                'structured_data': combined_data  # Store the structured JSON data
            }
            
            self.logger.info(f"Successfully parsed {file_path.name}: {len(combined_text)} characters extracted")
            return result
            
        except Exception as e:
            self.logger.error(f"OpenAI parsing failed for {file_path.name}: {e}")
            raise OpenAIAPIError(f"Failed to parse document: {e}")

    def _pdf_to_images(self, pdf_path: Path) -> list:
        """
        Convert PDF pages to images for vision processing.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of image file paths
        """
        try:
            doc = fitz.open(pdf_path)
            image_paths = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Render page to image with very high DPI for better handwriting recognition
                mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for better handwriting recognition
                pix = page.get_pixmap(matrix=mat)
                
                # Save as temporary image
                img_path = pdf_path.parent / f"{pdf_path.stem}_page_{page_num+1}.png"
                pix.save(img_path)
                image_paths.append(img_path)
            
            doc.close()
            return image_paths
            
        except Exception as e:
            raise OpenAIAPIError(f"Failed to convert PDF to images: {e}")

    def _encode_image_to_base64(self, image_path: Path) -> str:
        """
        Encode image to base64 for OpenAI API.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise OpenAIAPIError(f"Failed to encode image: {e}")

    def _call_openai_vision_api(self, image_base64: str, filename: str, page_num: int) -> str:
        """
        Call OpenAI Vision API to extract structured JSON data from image.
        
        Args:
            image_base64: Base64 encoded image
            filename: Original filename for context
            page_num: Page number being processed
            
        Returns:
            JSON string with structured data
        """
        # Enhanced prompt for crew feedback forms - handles multiple feedbacks and handwritten text
        system_prompt = """You are an expert at extracting structured data from crew feedback survey forms. You can handle:
        1. Multiple feedback forms in a single document
        2. Handwritten text and forms
        3. Mixed printed and handwritten content
        4. Partially filled or unclear forms
        
        IMPORTANT: If you detect MULTIPLE feedback forms in the same document, return an array of JSON objects.
        If there's only ONE feedback form, return a single JSON object.
        
        For SINGLE feedback form, use this structure:
        {
            "vessel": "vessel name",
            "crew_name": "full crew member name", 
            "crew_rank": "crew rank/position",
            "safer_with_sos": rating_number,
            "fatigue_monitoring_prevention": rating_number,
            "geofence_awareness": rating_number,
            "heat_stress_alerts_change": rating_number,
            "work_rest_hour_monitoring": rating_number,
            "ptw_system_improvement": rating_number,
            "paperwork_error_reduction": rating_number,
            "noise_exposure_monitoring": rating_number,
            "activity_tracking_awareness": rating_number,
            "fall_detection_confidence": rating_number,
            "feature_preference": "text description of preferred features",
            "additional_comments": "any other comments or feedback",
            "form_type": "single"
        }
        
        For MULTIPLE feedback forms, return:
        {
            "form_type": "multiple",
            "feedbacks": [
                {feedback_object_1},
                {feedback_object_2},
                ...
            ]
        }
        
        For ratings, use numbers 1-5 where:
        1 = Strongly Disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly Agree
        
        For handwritten text:
        - Do your best to read unclear handwriting
        - If text is completely illegible, use "illegible_handwriting" as the value
        - If partially readable, include what you can read with [unclear] for uncertain parts
        
        If a field is not found or unclear, use null for that field.
        Return ONLY valid JSON, no other text."""
        
        user_prompt = f"""Analyze this crew feedback survey form (page {page_num} of {filename}) and extract structured data as JSON.

CRITICAL: First determine if this document contains:
- ONE feedback form → return single JSON object with "form_type": "single"
- MULTIPLE feedback forms → return JSON with "form_type": "multiple" and "feedbacks" array

For each feedback form, look for:
1. Vessel name (usually at the top of each form)
2. Crew member's full name (may be handwritten)
3. Crew rank/position (like "2nd Officer", "Captain", "Able Seaman", etc.)
4. Survey ratings (look for checked boxes ☑, circled numbers, or written numbers 1-5)
5. Feature preferences (which SOL-X features are most valuable)
6. Any additional comments (may be handwritten)

HANDWRITING HANDLING:
- Read handwritten text carefully, even if messy
- For unclear handwriting, include what you can read + [unclear] for uncertain parts
- If completely illegible, use "illegible_handwriting"
- Look for common handwriting patterns in names and comments

RATING DETECTION:
- Look for checked boxes, circled numbers, highlighted options
- Check for handwritten numbers or marks next to questions
- If rating method is unclear, note the marking style in additional_comments

Map survey questions to these fields:
- "safer_with_sos": SOS/Crew Assist safety question
- "fatigue_monitoring_prevention": Fatigue monitoring question  
- "geofence_awareness": Geofence/boundary awareness question
- "heat_stress_alerts_change": Heat stress alerts question
- "work_rest_hour_monitoring": Work rest hour recording question
- "ptw_system_improvement": PTW (Permit to Work) system question
- "paperwork_error_reduction": Digital documentation/paperwork question
- "noise_exposure_monitoring": Noise exposure monitoring question
- "activity_tracking_awareness": Step count/activity tracking question
- "fall_detection_confidence": Fall detection safety question

Return ONLY the JSON object/array, no other text."""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result['choices'][0]['message']['content']
                
                self.logger.debug(f"OpenAI extraction successful for page {page_num}")
                return extracted_text
            else:
                error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise OpenAIAPIError(error_msg)
                
        except requests.exceptions.RequestException as e:
            raise OpenAIAPIError(f"Request failed: {e}")

    def _combine_page_data(self, all_data: list, filename: str) -> Dict[str, Any]:
        """
        Intelligently combine data from multiple pages, handling multiple feedbacks.
        
        Args:
            all_data: List of parsed data from each page
            filename: Original filename for logging
            
        Returns:
            Combined data structure
        """
        if len(all_data) == 1:
            return all_data[0]
        
        # Check if any page contains multiple feedbacks
        multiple_feedbacks = []
        single_feedbacks = []
        
        for page_data in all_data:
            if isinstance(page_data, dict):
                if page_data.get('form_type') == 'multiple':
                    # Extract individual feedbacks from multiple feedback pages
                    feedbacks = page_data.get('feedbacks', [])
                    multiple_feedbacks.extend(feedbacks)
                    self.logger.info(f"Found {len(feedbacks)} multiple feedbacks on page in {filename}")
                elif page_data.get('form_type') == 'single' or 'crew_name' in page_data:
                    # Single feedback form
                    single_feedbacks.append(page_data)
                else:
                    # Raw text or other format
                    single_feedbacks.append(page_data)
        
        # Combine all feedbacks
        all_feedbacks = multiple_feedbacks + single_feedbacks
        
        if len(all_feedbacks) > 1:
            self.logger.info(f"Combined {len(all_feedbacks)} total feedbacks from {filename}")
            return {
                'form_type': 'multiple',
                'feedbacks': all_feedbacks,
                'total_feedbacks': len(all_feedbacks)
            }
        elif len(all_feedbacks) == 1:
            # Single feedback, but mark it as processed from multi-page
            feedback = all_feedbacks[0]
            feedback['form_type'] = 'single'
            return feedback
        else:
            # Fallback: merge pages traditionally
            self.logger.warning(f"No clear feedback structure found in {filename}, using traditional merge")
            combined_data = {}
            for page_data in all_data:
                if isinstance(page_data, dict):
                    for key, value in page_data.items():
                        if value is not None and (key not in combined_data or combined_data[key] is None):
                            combined_data[key] = value
            return combined_data

    def _convert_json_to_text(self, data: Dict[str, Any]) -> str:
        """
        Convert structured JSON data back to text format for compatibility.
        Handles both single and multiple feedback forms.
        
        Args:
            data: Structured data dictionary
            
        Returns:
            Text representation of the data
        """
        if 'raw_text' in data:
            return data['raw_text']
        
        # Handle multiple feedbacks
        if data.get('form_type') == 'multiple':
            text_parts = [f"MULTIPLE FEEDBACKS DETECTED ({data.get('total_feedbacks', 0)} forms)"]
            text_parts.append("=" * 60)
            
            feedbacks = data.get('feedbacks', [])
            for i, feedback in enumerate(feedbacks, 1):
                text_parts.append(f"\n--- FEEDBACK {i} ---")
                text_parts.append(self._convert_single_feedback_to_text(feedback))
                text_parts.append("-" * 40)
            
            return '\n'.join(text_parts)
        
        # Handle single feedback
        return self._convert_single_feedback_to_text(data)

    def _convert_single_feedback_to_text(self, data: Dict[str, Any]) -> str:
        """
        Convert a single feedback form to text format.
        
        Args:
            data: Single feedback data dictionary
            
        Returns:
            Text representation of the single feedback
        """
        text_parts = []
        
        # Basic info
        if data.get('vessel'):
            text_parts.append(f"Vessel: {data['vessel']}")
        if data.get('crew_name'):
            text_parts.append(f"Crew Name: {data['crew_name']}")
        if data.get('crew_rank'):
            text_parts.append(f"Crew Rank: {data['crew_rank']}")
        
        # Ratings
        rating_fields = [
            ('safer_with_sos', 'Safer with SOS'),
            ('fatigue_monitoring_prevention', 'Fatigue Monitoring Prevention'),
            ('geofence_awareness', 'Geofence Awareness'),
            ('heat_stress_alerts_change', 'Heat Stress Alerts Change'),
            ('work_rest_hour_monitoring', 'Work Rest Hour Monitoring'),
            ('ptw_system_improvement', 'PTW System Improvement'),
            ('paperwork_error_reduction', 'Paperwork Error Reduction'),
            ('noise_exposure_monitoring', 'Noise Exposure Monitoring'),
            ('activity_tracking_awareness', 'Activity Tracking Awareness'),
            ('fall_detection_confidence', 'Fall Detection Confidence')
        ]
        
        text_parts.append("\nRatings:")
        for field_key, field_name in rating_fields:
            rating = data.get(field_key)
            if rating is not None:
                text_parts.append(f"{field_name}: {rating}")
        
        # Additional info
        if data.get('feature_preference'):
            text_parts.append(f"\nFeature Preference: {data['feature_preference']}")
        if data.get('additional_comments'):
            text_parts.append(f"Additional Comments: {data['additional_comments']}")
        
        return '\n'.join(text_parts)

    def validate_connection(self) -> bool:
        """
        Test OpenAI API connection and authentication.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Simple API test
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("OpenAI API connection validated successfully")
                return True
            else:
                self.logger.error(f"OpenAI API connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"OpenAI API connection validation failed: {e}")
            return False