"""
Data extraction service for parsing crew feedback forms.
Extracts structured data from parsed text using pattern matching and NLP techniques.
"""
import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from ..models.feedback_data import FeedbackData
from ..config.config_manager import ConfigManager


@dataclass
class ExtractionResult:
    """Result of data extraction with confidence scoring."""
    data: Optional[FeedbackData]
    confidence_score: float  # 0.0 to 1.0
    missing_fields: List[str]
    extraction_notes: List[str]


class DataExtractor:
    """
    Service for extracting structured data from parsed crew feedback forms.
    Uses pattern matching and keyword detection to identify crew information,
    ratings, and preferences from text.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None, use_openai: bool = True):
        """
        Initialize data extractor with logging and pattern compilation.
        
        Args:
            config_manager: Configuration manager for OpenAI integration
            use_openai: Whether to use OpenAI for enhanced extraction
        """
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()
        
        # Initialize OpenAI client if available and requested
        self.openai_client = None
        self.use_openai = use_openai
        
        if use_openai and config_manager:
            try:
                from .openai_client import OpenAIClient
                self.openai_client = OpenAIClient(config_manager)
                self.logger.info("OpenAI integration enabled for enhanced extraction")
            except Exception as e:
                self.logger.warning(f"OpenAI integration not available: {e}")
                self.use_openai = False
        
        # Rating field mappings for extraction
        self.rating_fields = {
            'safer_with_sos': ['safer with sos', 'sos safety', 'sos system', 'safer sos'],
            'fatigue_monitoring_prevention': ['fatigue monitoring', 'fatigue prevention', 'fatigue management', 'fatigue alert'],
            'geofence_awareness': ['geofence', 'geo fence', 'boundary awareness', 'location awareness'],
            'heat_stress_alerts_change': ['heat stress', 'heat alert', 'temperature monitoring', 'thermal stress'],
            'work_rest_hour_monitoring': ['work rest', 'rest hour', 'working hours', 'rest monitoring'],
            'ptw_system_improvement': ['ptw', 'permit to work', 'work permit', 'ptw system'],
            'paperwork_error_reduction': ['paperwork error', 'documentation error', 'paper error', 'admin error'],
            'noise_exposure_monitoring': ['noise exposure', 'noise monitoring', 'sound level', 'noise level'],
            'activity_tracking_awareness': ['activity tracking', 'activity monitoring', 'movement tracking', 'activity awareness'],
            'fall_detection_confidence': ['fall detection', 'fall alert', 'fall monitoring', 'fall prevention']
        }

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient text matching."""
        # Vessel name patterns - improved for SOL-X format
        self.vessel_patterns = [
            re.compile(r'vessel\s*:?[_\s]*\n?\s*([A-Za-z0-9\s\-_]+)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'ship\s*:?[_\s]*\n?\s*([A-Za-z0-9\s\-_]+)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'vessel\s+name\s*:?[_\s]*\n?\s*([A-Za-z0-9\s\-_]+)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'ship\s+name\s*:?[_\s]*\n?\s*([A-Za-z0-9\s\-_]+)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'mv\s+([A-Za-z0-9\s\-_]+)', re.IGNORECASE),
            re.compile(r'm\.?v\.?\s+([A-Za-z0-9\s\-_]+)', re.IGNORECASE)
        ]
        
        # Crew name patterns - improved for SOL-X format
        self.crew_name_patterns = [
            re.compile(r'crew\s+name\s*:?[_\s]*\n?\s*([A-Za-z\s\-\'\.]+)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'name\s*:?[_\s]*\n?\s*([A-Za-z\s\-\'\.]+)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'employee\s+name\s*:?[_\s]*\n?\s*([A-Za-z\s\-\'\.]+)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'full\s+name\s*:?[_\s]*\n?\s*([A-Za-z\s\-\'\.]+)', re.IGNORECASE | re.MULTILINE),
            # Specific pattern for SOL-X format with underscores and line breaks
            re.compile(r'crew\s+name\s*:?[_\s]*\n\s*([A-Z\s]+(?:\s+BIN)?)', re.IGNORECASE | re.MULTILINE)
        ]
        
        # Crew rank patterns - improved for SOL-X format
        self.crew_rank_patterns = [
            re.compile(r'crew\s+rank\s*:?[_\s]*\n?\s*([A-Za-z0-9\s\-]+)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'rank\s*:?[_\s]*\n?\s*([A-Za-z0-9\s\-]+)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'position\s*:?\s*([A-Za-z\s\-]+)', re.IGNORECASE),
            re.compile(r'job\s+title\s*:?\s*([A-Za-z\s\-]+)', re.IGNORECASE),
            re.compile(r'crew\s+rank\s*:?\s*([A-Za-z\s\-]+)', re.IGNORECASE),
            re.compile(r'designation\s*:?\s*([A-Za-z\s\-]+)', re.IGNORECASE)
        ]
        
        # Rating patterns (1-5 scale)
        self.rating_patterns = [
            re.compile(r'(\d)\s*/\s*5', re.IGNORECASE),  # "3/5" format
            re.compile(r'rating\s*:?\s*(\d)', re.IGNORECASE),  # "rating: 4"
            re.compile(r'score\s*:?\s*(\d)', re.IGNORECASE),  # "score: 3"
            re.compile(r'(\d)\s+out\s+of\s+5', re.IGNORECASE),  # "4 out of 5"
            re.compile(r'(\d)\s*\*+', re.IGNORECASE),  # "3***" star format
            re.compile(r'[^\d](\d)[^\d]', re.IGNORECASE)  # Single digit surrounded by non-digits
        ]

    def extract_crew_information(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract vessel name, crew name, and crew rank from parsed text.
        
        Args:
            text: Parsed text from document
            
        Returns:
            Tuple of (vessel_name, crew_name, crew_rank)
        """
        self.logger.debug("Extracting crew information from text")
        
        vessel_name = self._extract_vessel_name(text)
        crew_name = self._extract_crew_name(text)
        crew_rank = self._extract_crew_rank(text)
        
        self.logger.debug(f"Extracted crew info - Vessel: {vessel_name}, Name: {crew_name}, Rank: {crew_rank}")
        
        return vessel_name, crew_name, crew_rank

    def _extract_vessel_name(self, text: str) -> Optional[str]:
        """Extract vessel name using pattern matching."""
        for pattern in self.vessel_patterns:
            match = pattern.search(text)
            if match:
                vessel_name = match.group(1).strip()
                # Clean up common artifacts
                vessel_name = re.sub(r'\s+', ' ', vessel_name)  # Multiple spaces to single
                vessel_name = vessel_name.strip('.,;:-_')  # Remove trailing punctuation
                
                # Remove form artifacts that get mixed in
                vessel_name = re.sub(r'\bCrew\s+Name\b', '', vessel_name, flags=re.IGNORECASE).strip()
                vessel_name = re.sub(r'^[_\-\s]+|[_\-\s]+$', '', vessel_name).strip()
                
                if (len(vessel_name) > 2 and 
                    not vessel_name.isdigit() and
                    not vessel_name.lower() in ['vessel', 'ship', 'name']):
                    return vessel_name
        
        return None

    def _extract_crew_name(self, text: str) -> Optional[str]:
        """Extract crew member name using pattern matching."""
        for pattern in self.crew_name_patterns:
            match = pattern.search(text)
            if match:
                crew_name = match.group(1).strip()
                # Clean up common artifacts
                crew_name = re.sub(r'\s+', ' ', crew_name)  # Multiple spaces to single
                crew_name = crew_name.strip('.,;:-_')  # Remove trailing punctuation
                
                # Remove common form artifacts that get mixed in
                crew_name = re.sub(r'\bCrew\s+Name\b', '', crew_name, flags=re.IGNORECASE).strip()
                crew_name = re.sub(r'\bCrew\s+Rank\b', '', crew_name, flags=re.IGNORECASE).strip()
                crew_name = re.sub(r'\bPart\s+[A-Z]\b', '', crew_name, flags=re.IGNORECASE).strip()
                crew_name = re.sub(r'^[_\-\s]+|[_\-\s]+$', '', crew_name).strip()
                
                # Validate name (should contain letters and be reasonable length)
                if (len(crew_name) > 2 and 
                    re.search(r'[A-Za-z]', crew_name) and 
                    len(crew_name) < 50 and
                    not crew_name.lower() in ['crew name', 'name', 'crew']):
                    return crew_name
        
        return None

    def _extract_crew_rank(self, text: str) -> Optional[str]:
        """Extract crew rank/position using pattern matching."""
        # Common maritime ranks for validation
        common_ranks = [
            'captain', 'chief officer', 'second officer', 'third officer',
            'chief engineer', 'second engineer', 'third engineer',
            'bosun', 'able seaman', 'ordinary seaman', 'deck hand',
            'cook', 'steward', 'oiler', 'wiper', 'fitter',
            'electrician', 'radio officer', 'purser'
        ]
        
        for pattern in self.crew_rank_patterns:
            match = pattern.search(text)
            if match:
                crew_rank = match.group(1).strip()
                # Clean up common artifacts
                crew_rank = re.sub(r'\s+', ' ', crew_rank)  # Multiple spaces to single
                crew_rank = crew_rank.strip('.,;:-_')  # Remove trailing punctuation
                crew_rank = crew_rank.lower()
                
                # Validate rank (should be reasonable length and contain letters)
                if (len(crew_rank) > 2 and 
                    re.search(r'[A-Za-z]', crew_rank) and 
                    len(crew_rank) < 30):
                    
                    # Check if it matches common ranks or contains rank-like words
                    if (crew_rank in common_ranks or 
                        any(rank_word in crew_rank for rank_word in ['officer', 'engineer', 'seaman', 'chief', 'captain'])):
                        return crew_rank.title()  # Return in title case
                    
                    # If not a common rank but looks valid, return it anyway
                    return crew_rank.title()
        
        return None

    def extract_ratings(self, text: str) -> Dict[str, Optional[int]]:
        """
        Extract all 10 rating metrics (1-5 scale) from text.
        Handles various text formats and number representations.
        
        Args:
            text: Parsed text from document
            
        Returns:
            Dict mapping rating field names to extracted values (1-5 or None)
        """
        self.logger.debug("Extracting rating metrics from text")
        
        ratings = {}
        text_lower = text.lower()
        
        for field_name, keywords in self.rating_fields.items():
            rating = self._extract_single_rating(text_lower, keywords)
            ratings[field_name] = rating
            
        self.logger.debug(f"Extracted ratings: {ratings}")
        return ratings

    def _extract_single_rating(self, text: str, keywords: List[str]) -> Optional[int]:
        """
        Extract a single rating value for given keywords.
        
        Args:
            text: Lowercase text to search
            keywords: List of keywords to look for
            
        Returns:
            Rating value (1-5) or None if not found
        """
        best_rating = None
        best_confidence = 0
        
        for keyword in keywords:
            # Find all occurrences of the keyword
            keyword_positions = []
            start = 0
            while True:
                pos = text.find(keyword, start)
                if pos == -1:
                    break
                keyword_positions.append(pos)
                start = pos + 1
            
            # For each keyword occurrence, look for nearby ratings
            for pos in keyword_positions:
                # Define search window around keyword (±100 characters)
                window_start = max(0, pos - 100)
                window_end = min(len(text), pos + len(keyword) + 100)
                window_text = text[window_start:window_end]
                
                # Try different rating extraction methods
                rating = self._find_rating_in_window(window_text, keyword)
                
                if rating is not None:
                    # Calculate confidence based on proximity to keyword
                    distance = abs(pos - (window_start + window_text.find(str(rating))))
                    confidence = max(0, 1.0 - (distance / 100.0))  # Closer = higher confidence
                    
                    if confidence > best_confidence:
                        best_rating = rating
                        best_confidence = confidence
        
        return best_rating

    def _find_rating_in_window(self, window_text: str, keyword: str) -> Optional[int]:
        """
        Find rating value within a text window using various patterns.
        
        Args:
            window_text: Text window to search
            keyword: The keyword being searched for
            
        Returns:
            Rating value (1-5) or None
        """
        # Method 1: Look for explicit rating patterns
        for pattern in self.rating_patterns:
            matches = pattern.findall(window_text)
            for match in matches:
                try:
                    rating = int(match)
                    if 1 <= rating <= 5:
                        return rating
                except (ValueError, TypeError):
                    continue
        
        # Method 2: Look for checkbox/selection patterns
        checkbox_patterns = [
            r'[x✓✗]\s*([1-5])',  # [x] 4, ✓ 3, etc.
            r'([1-5])\s*[x✓✗]',  # 4 [x], 3 ✓, etc.
            r'●\s*([1-5])',      # ● 4 (filled circle)
            r'([1-5])\s*●',      # 4 ● 
        ]
        
        for pattern_str in checkbox_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            matches = pattern.findall(window_text)
            for match in matches:
                try:
                    rating = int(match)
                    if 1 <= rating <= 5:
                        return rating
                except (ValueError, TypeError):
                    continue
        
        # Method 3: Look for word-based ratings
        word_ratings = {
            'excellent': 5, 'outstanding': 5, 'exceptional': 5,
            'very good': 4, 'good': 4, 'satisfactory': 4,
            'average': 3, 'fair': 3, 'moderate': 3, 'okay': 3,
            'poor': 2, 'below average': 2, 'unsatisfactory': 2,
            'very poor': 1, 'terrible': 1, 'unacceptable': 1
        }
        
        for word, rating in word_ratings.items():
            if word in window_text:
                return rating
        
        # Method 4: Look for scale indicators
        scale_patterns = [
            r'strongly\s+agree',     # 5
            r'agree',                # 4  
            r'neutral',              # 3
            r'disagree',             # 2
            r'strongly\s+disagree'   # 1
        ]
        
        scale_values = [5, 4, 3, 2, 1]
        for i, pattern_str in enumerate(scale_patterns):
            if re.search(pattern_str, window_text, re.IGNORECASE):
                return scale_values[i]
        
        return None

    def _extract_numeric_ratings_from_text(self, text: str) -> List[int]:
        """
        Extract all numeric ratings (1-5) from text as fallback method.
        
        Args:
            text: Text to search
            
        Returns:
            List of valid ratings found
        """
        ratings = []
        
        # Find all single digits that could be ratings
        digit_pattern = re.compile(r'\b([1-5])\b')
        matches = digit_pattern.findall(text)
        
        for match in matches:
            try:
                rating = int(match)
                if 1 <= rating <= 5:
                    ratings.append(rating)
            except (ValueError, TypeError):
                continue
        
        return ratings

    def extract_feature_preference(self, text: str) -> Optional[str]:
        """
        Extract feature preference text or selection from forms.
        Handles both text responses and checkbox selections.
        
        Args:
            text: Parsed text from document
            
        Returns:
            Feature preference text or None if not found
        """
        self.logger.debug("Extracting feature preference from text")
        
        # Try different extraction methods
        preference = (self._extract_preference_from_questions(text) or
                     self._extract_preference_from_checkboxes(text) or
                     self._extract_preference_from_sections(text))
        
        if preference:
            # Clean up the extracted preference
            preference = self._clean_preference_text(preference)
            
        self.logger.debug(f"Extracted feature preference: {preference}")
        return preference

    def _extract_preference_from_questions(self, text: str) -> Optional[str]:
        """Extract preference from question-answer patterns."""
        preference_patterns = [
            re.compile(r'feature\s+preference\s*:?\s*([^\n\r]+)', re.IGNORECASE),
            re.compile(r'preferred\s+feature\s*:?\s*([^\n\r]+)', re.IGNORECASE),
            re.compile(r'which\s+feature.*?\?\s*([^\n\r]+)', re.IGNORECASE),
            re.compile(r'favorite\s+feature\s*:?\s*([^\n\r]+)', re.IGNORECASE),
            re.compile(r'most\s+important\s+feature\s*:?\s*([^\n\r]+)', re.IGNORECASE),
            re.compile(r'what\s+feature.*?\?\s*([^\n\r]+)', re.IGNORECASE),
            re.compile(r'additional\s+features?\s*:?\s*([^\n\r]+)', re.IGNORECASE),
            re.compile(r'suggestions?\s*:?\s*([^\n\r]+)', re.IGNORECASE)
        ]
        
        for pattern in preference_patterns:
            match = pattern.search(text)
            if match:
                preference = match.group(1).strip()
                if len(preference) > 3 and not preference.isdigit():
                    return preference
        
        return None

    def _extract_preference_from_checkboxes(self, text: str) -> Optional[str]:
        """Extract preference from checkbox selections."""
        # Common feature options that might appear as checkboxes
        feature_options = [
            'fatigue monitoring', 'heat stress alerts', 'fall detection',
            'noise monitoring', 'activity tracking', 'geofence alerts',
            'work rest monitoring', 'sos system', 'ptw system',
            'paperwork reduction', 'safety alerts', 'emergency response',
            'communication system', 'training modules', 'reporting system'
        ]
        
        selected_features = []
        
        # Look for checked boxes or selected items
        checkbox_patterns = [
            r'[x✓✗]\s*([^\n\r]+)',  # [x] feature name
            r'●\s*([^\n\r]+)',      # ● feature name (filled circle)
            r'selected\s*:?\s*([^\n\r]+)',  # selected: feature name
            r'chosen\s*:?\s*([^\n\r]+)'     # chosen: feature name
        ]
        
        for pattern_str in checkbox_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            matches = pattern.findall(text)
            
            for match in matches:
                match_lower = match.lower().strip()
                # Check if this matches any known feature
                for feature in feature_options:
                    if feature in match_lower or any(word in match_lower for word in feature.split()):
                        if match.strip() not in selected_features:
                            selected_features.append(match.strip())
        
        if selected_features:
            return ', '.join(selected_features)
        
        return None

    def _extract_preference_from_sections(self, text: str) -> Optional[str]:
        """Extract preference from dedicated sections or comments."""
        # Look for sections that might contain preferences
        section_patterns = [
            re.compile(r'comments?\s*:?\s*([^\n\r]{10,})', re.IGNORECASE),
            re.compile(r'feedback\s*:?\s*([^\n\r]{10,})', re.IGNORECASE),
            re.compile(r'suggestions?\s*:?\s*([^\n\r]{10,})', re.IGNORECASE),
            re.compile(r'recommendations?\s*:?\s*([^\n\r]{10,})', re.IGNORECASE),
            re.compile(r'improvements?\s*:?\s*([^\n\r]{10,})', re.IGNORECASE),
            re.compile(r'additional\s+thoughts?\s*:?\s*([^\n\r]{10,})', re.IGNORECASE)
        ]
        
        for pattern in section_patterns:
            match = pattern.search(text)
            if match:
                preference = match.group(1).strip()
                # Filter out common non-preference text
                if (len(preference) > 10 and 
                    not preference.lower().startswith(('none', 'n/a', 'not applicable', 'no comment'))):
                    return preference
        
        # Look for multi-line preference sections
        lines = text.split('\n')
        preference_section = []
        in_preference_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a preference section
            if any(keyword in line.lower() for keyword in 
                   ['preference', 'suggest', 'recommend', 'improve', 'feature', 'comment']):
                in_preference_section = True
                # Extract text after the keyword
                for keyword in ['preference:', 'suggestions:', 'comments:', 'feedback:']:
                    if keyword in line.lower():
                        after_keyword = line.lower().split(keyword, 1)
                        if len(after_keyword) > 1 and after_keyword[1].strip():
                            preference_section.append(after_keyword[1].strip())
                        break
                continue
            
            # If we're in a preference section, collect lines until we hit a new section
            if in_preference_section:
                if (line.lower().startswith(('name:', 'rank:', 'vessel:', 'date:')) or
                    any(rating_keyword in line.lower() for rating_keyword in 
                        ['rating', 'score', '/5', 'out of'])):
                    break
                preference_section.append(line)
        
        if preference_section:
            combined_preference = ' '.join(preference_section).strip()
            if len(combined_preference) > 10:
                return combined_preference
        
        return None

    def _clean_preference_text(self, preference: str) -> str:
        """Clean and normalize preference text."""
        if not preference:
            return ""
        
        # Remove common artifacts
        preference = re.sub(r'\s+', ' ', preference)  # Multiple spaces to single
        preference = preference.strip('.,;:-_')  # Remove trailing punctuation
        
        # Remove common prefixes
        prefixes_to_remove = [
            'preference:', 'preferred:', 'feature:', 'suggestion:', 
            'comment:', 'feedback:', 'recommendation:'
        ]
        
        for prefix in prefixes_to_remove:
            if preference.lower().startswith(prefix):
                preference = preference[len(prefix):].strip()
        
        # Capitalize first letter
        if preference:
            preference = preference[0].upper() + preference[1:]
        
        return preference

    def extract_complete_data(self, text: str, filename: str = "unknown") -> ExtractionResult:
        """
        Extract complete feedback data with optional OpenAI enhancement.
        
        Args:
            text: Parsed text from document
            filename: Source filename for logging
            
        Returns:
            ExtractionResult with extracted data and confidence score
        """
        # Try OpenAI extraction first if available
        if self.use_openai and self.openai_client:
            try:
                return self._extract_with_openai(text, filename)
            except Exception as e:
                self.logger.warning(f"OpenAI extraction failed for {filename}, falling back to pattern matching: {e}")
        
        # Fall back to pattern matching extraction
        return self._extract_with_patterns(text)

    def _extract_with_openai(self, text: str, filename: str) -> ExtractionResult:
        """
        Extract data using OpenAI GPT model.
        
        Args:
            text: Parsed text from document
            filename: Source filename for logging
            
        Returns:
            ExtractionResult with extracted data
        """
        self.logger.info(f"Using OpenAI extraction for {filename}")
        
        # Get structured data from OpenAI
        structured_data = self.openai_client.extract_structured_data(text, filename)
        validated_data = self.openai_client.validate_extracted_data(structured_data)
        
        # Convert to FeedbackData object
        feedback_data = FeedbackData(
            vessel=validated_data.get('vessel'),
            crew_name=validated_data.get('crew_name'),
            crew_rank=validated_data.get('crew_rank'),
            safer_with_sos=validated_data.get('safer_with_sos'),
            fatigue_monitoring_prevention=validated_data.get('fatigue_monitoring_prevention'),
            geofence_awareness=validated_data.get('geofence_awareness'),
            heat_stress_alerts_change=validated_data.get('heat_stress_alerts_change'),
            work_rest_hour_monitoring=validated_data.get('work_rest_hour_monitoring'),
            ptw_system_improvement=validated_data.get('ptw_system_improvement'),
            paperwork_error_reduction=validated_data.get('paperwork_error_reduction'),
            noise_exposure_monitoring=validated_data.get('noise_exposure_monitoring'),
            activity_tracking_awareness=validated_data.get('activity_tracking_awareness'),
            fall_detection_confidence=validated_data.get('fall_detection_confidence'),
            feature_preference=validated_data.get('feature_preference'),
            additional_comments=validated_data.get('additional_comments')
        )
        
        # Calculate confidence and missing fields
        confidence_score = self.openai_client.get_extraction_confidence(validated_data)
        missing_fields = [field for field, value in validated_data.items() if value is None]
        
        # Validate the extracted data
        validation_result = self.validate_extracted_data(feedback_data)
        extraction_notes = []
        
        if not validation_result['is_valid']:
            extraction_notes.extend(validation_result['errors'])
        
        # Add confidence notes
        if confidence_score < 0.5:
            extraction_notes.append("Low confidence extraction - manual review recommended")
        elif confidence_score < 0.7:
            extraction_notes.append("Medium confidence extraction - verification recommended")
        
        extraction_notes.append("Extracted using OpenAI GPT model")
        
        self.logger.info(f"OpenAI extraction completed - Confidence: {confidence_score:.2f}, Missing fields: {len(missing_fields)}")
        
        return ExtractionResult(
            data=feedback_data,
            confidence_score=confidence_score,
            missing_fields=missing_fields,
            extraction_notes=extraction_notes
        )

    def _extract_with_patterns(self, text: str) -> ExtractionResult:
        """
        Extract complete feedback data with validation and quality scoring.
        
        Args:
            text: Parsed text from document
            
        Returns:
            ExtractionResult with data, confidence score, and validation info
        """
        self.logger.debug("Starting complete data extraction")
        
        extraction_notes = []
        missing_fields = []
        
        # Extract crew information
        vessel_name, crew_name, crew_rank = self.extract_crew_information(text)
        
        # Extract ratings
        ratings = self.extract_ratings(text)
        
        # Extract feature preference
        feature_preference = self.extract_feature_preference(text)
        
        # Validate and create FeedbackData object
        try:
            feedback_data = FeedbackData(
                vessel=vessel_name or "",
                crew_name=crew_name or "",
                crew_rank=crew_rank or "",
                safer_with_sos=ratings.get('safer_with_sos'),
                fatigue_monitoring_prevention=ratings.get('fatigue_monitoring_prevention'),
                geofence_awareness=ratings.get('geofence_awareness'),
                heat_stress_alerts_change=ratings.get('heat_stress_alerts_change'),
                work_rest_hour_monitoring=ratings.get('work_rest_hour_monitoring'),
                ptw_system_improvement=ratings.get('ptw_system_improvement'),
                paperwork_error_reduction=ratings.get('paperwork_error_reduction'),
                noise_exposure_monitoring=ratings.get('noise_exposure_monitoring'),
                activity_tracking_awareness=ratings.get('activity_tracking_awareness'),
                fall_detection_confidence=ratings.get('fall_detection_confidence'),
                feature_preference=feature_preference or ""
            )
        except Exception as e:
            self.logger.error(f"Failed to create FeedbackData object: {e}")
            return ExtractionResult(
                data=None,
                confidence_score=0.0,
                missing_fields=[],
                extraction_notes=[f"Failed to create data object: {e}"]
            )
        
        # Validate extracted data
        validation_result = self.validate_extracted_data(feedback_data)
        
        # Calculate confidence score
        confidence_score = self.calculate_confidence_score(feedback_data, text)
        
        # Collect missing fields
        if not vessel_name:
            missing_fields.append('vessel')
        if not crew_name:
            missing_fields.append('crew_name')
        if not crew_rank:
            missing_fields.append('crew_rank')
        if not feature_preference:
            missing_fields.append('feature_preference')
        
        # Check for missing ratings
        for field_name, rating in ratings.items():
            if rating is None:
                missing_fields.append(field_name)
        
        # Add validation notes
        if not validation_result['is_valid']:
            extraction_notes.extend(validation_result['errors'])
        
        # Add quality notes
        if confidence_score < 0.5:
            extraction_notes.append("Low confidence extraction - manual review recommended")
        elif confidence_score < 0.7:
            extraction_notes.append("Medium confidence extraction - verification recommended")
        
        self.logger.info(f"Data extraction completed - Confidence: {confidence_score:.2f}, Missing fields: {len(missing_fields)}")
        
        return ExtractionResult(
            data=feedback_data,
            confidence_score=confidence_score,
            missing_fields=missing_fields,
            extraction_notes=extraction_notes
        )

    def validate_extracted_data(self, data: FeedbackData) -> Dict[str, Any]:
        """
        Validate extracted data completeness and accuracy.
        
        Args:
            data: FeedbackData object to validate
            
        Returns:
            Dict with validation results
        """
        errors = []
        warnings = []
        
        # Validate crew information
        if not data.vessel or len(data.vessel.strip()) < 2:
            errors.append("Vessel name is missing or too short")
        
        if not data.crew_name or len(data.crew_name.strip()) < 2:
            errors.append("Crew name is missing or too short")
        
        if not data.crew_rank or len(data.crew_rank.strip()) < 2:
            errors.append("Crew rank is missing or too short")
        
        # Validate ratings using the model's built-in validation
        if not data.validate_ratings():
            invalid_ratings = data.get_invalid_ratings()
            errors.append(f"Invalid ratings found: {invalid_ratings}")
        
        # Check for missing ratings
        rating_fields = [
            'safer_with_sos', 'fatigue_monitoring_prevention', 'geofence_awareness',
            'heat_stress_alerts_change', 'work_rest_hour_monitoring', 'ptw_system_improvement',
            'paperwork_error_reduction', 'noise_exposure_monitoring', 
            'activity_tracking_awareness', 'fall_detection_confidence'
        ]
        
        missing_ratings = []
        for field in rating_fields:
            if getattr(data, field) is None:
                missing_ratings.append(field)
        
        if missing_ratings:
            warnings.append(f"Missing ratings: {missing_ratings}")
        
        # Validate feature preference
        if not data.feature_preference or len(data.feature_preference.strip()) < 3:
            warnings.append("Feature preference is missing or too short")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'completeness_score': self._calculate_completeness_score(data)
        }

    def _calculate_completeness_score(self, data: FeedbackData) -> float:
        """Calculate completeness score (0.0 to 1.0) based on populated fields."""
        total_fields = 14  # Total number of fields in FeedbackData
        populated_fields = 0
        
        # Check crew information fields
        if data.vessel and len(data.vessel.strip()) >= 2:
            populated_fields += 1
        if data.crew_name and len(data.crew_name.strip()) >= 2:
            populated_fields += 1
        if data.crew_rank and len(data.crew_rank.strip()) >= 2:
            populated_fields += 1
        
        # Check rating fields
        rating_fields = [
            data.safer_with_sos, data.fatigue_monitoring_prevention, data.geofence_awareness,
            data.heat_stress_alerts_change, data.work_rest_hour_monitoring, data.ptw_system_improvement,
            data.paperwork_error_reduction, data.noise_exposure_monitoring,
            data.activity_tracking_awareness, data.fall_detection_confidence
        ]
        
        for rating in rating_fields:
            if rating is not None and 1 <= rating <= 5:
                populated_fields += 1
        
        # Check feature preference
        if data.feature_preference and len(data.feature_preference.strip()) >= 3:
            populated_fields += 1
        
        return populated_fields / total_fields

    def calculate_confidence_score(self, data: FeedbackData, original_text: str) -> float:
        """
        Calculate confidence score for extracted data based on multiple factors.
        
        Args:
            data: Extracted FeedbackData
            original_text: Original parsed text
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        scores = []
        
        # Completeness score (30% weight)
        completeness = self._calculate_completeness_score(data)
        scores.append(('completeness', completeness, 0.3))
        
        # Text quality score (20% weight)
        text_quality = self._assess_text_quality(original_text)
        scores.append(('text_quality', text_quality, 0.2))
        
        # Data consistency score (25% weight)
        consistency = self._assess_data_consistency(data, original_text)
        scores.append(('consistency', consistency, 0.25))
        
        # Pattern match confidence (25% weight)
        pattern_confidence = self._assess_pattern_match_confidence(data, original_text)
        scores.append(('pattern_match', pattern_confidence, 0.25))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        
        self.logger.debug(f"Confidence breakdown: {[(name, f'{score:.2f}') for name, score, _ in scores]}")
        
        return min(1.0, max(0.0, total_score))

    def _assess_text_quality(self, text: str) -> float:
        """Assess the quality of the parsed text."""
        if not text or len(text.strip()) < 50:
            return 0.1
        
        # Check for common OCR artifacts
        ocr_artifacts = ['|||', '###', '???', '***', '...', '___']
        artifact_count = sum(text.count(artifact) for artifact in ocr_artifacts)
        
        # Check character diversity (good text has varied characters)
        unique_chars = len(set(text.lower()))
        char_diversity = min(1.0, unique_chars / 50.0)
        
        # Check for reasonable word count
        words = text.split()
        word_count_score = min(1.0, len(words) / 100.0)
        
        # Penalize excessive artifacts
        artifact_penalty = max(0.0, 1.0 - (artifact_count / 10.0))
        
        return (char_diversity + word_count_score + artifact_penalty) / 3.0

    def _assess_data_consistency(self, data: FeedbackData, text: str) -> float:
        """Assess consistency between extracted data and source text."""
        consistency_score = 0.0
        checks = 0
        
        # Check if vessel name appears in text
        if data.vessel:
            checks += 1
            if data.vessel.lower() in text.lower():
                consistency_score += 1.0
            elif any(word in text.lower() for word in data.vessel.lower().split()):
                consistency_score += 0.5
        
        # Check if crew name appears in text
        if data.crew_name:
            checks += 1
            if data.crew_name.lower() in text.lower():
                consistency_score += 1.0
            elif any(word in text.lower() for word in data.crew_name.lower().split() if len(word) > 2):
                consistency_score += 0.5
        
        # Check if crew rank appears in text
        if data.crew_rank:
            checks += 1
            if data.crew_rank.lower() in text.lower():
                consistency_score += 1.0
            elif any(word in text.lower() for word in data.crew_rank.lower().split()):
                consistency_score += 0.5
        
        # Check if feature preference appears in text
        if data.feature_preference:
            checks += 1
            if data.feature_preference.lower() in text.lower():
                consistency_score += 1.0
            elif any(word in text.lower() for word in data.feature_preference.lower().split() if len(word) > 3):
                consistency_score += 0.5
        
        return consistency_score / max(1, checks)

    def _assess_pattern_match_confidence(self, data: FeedbackData, text: str) -> float:
        """Assess confidence based on pattern matching success."""
        # Count successful extractions
        successful_extractions = 0
        total_possible = 14  # Total fields in FeedbackData
        
        if data.vessel:
            successful_extractions += 1
        if data.crew_name:
            successful_extractions += 1
        if data.crew_rank:
            successful_extractions += 1
        if data.feature_preference:
            successful_extractions += 1
        
        # Count valid ratings
        rating_fields = [
            data.safer_with_sos, data.fatigue_monitoring_prevention, data.geofence_awareness,
            data.heat_stress_alerts_change, data.work_rest_hour_monitoring, data.ptw_system_improvement,
            data.paperwork_error_reduction, data.noise_exposure_monitoring,
            data.activity_tracking_awareness, data.fall_detection_confidence
        ]
        
        for rating in rating_fields:
            if rating is not None and 1 <= rating <= 5:
                successful_extractions += 1
        
        return successful_extractions / total_possible

    def flag_for_manual_review(self, extraction_result: ExtractionResult) -> bool:
        """
        Determine if a record should be flagged for manual review.
        
        Args:
            extraction_result: Result of data extraction
            
        Returns:
            bool: True if manual review is recommended
        """
        # Flag if confidence is very low
        if extraction_result.confidence_score < 0.3:
            return True
        
        # Flag if most fields are missing
        if len(extraction_result.missing_fields) > 10:  # More than 2/3 of the fields
            return True
        
        # Flag if ALL critical fields are missing (more lenient)
        critical_fields = ['vessel']  # Only vessel is truly critical
        missing_critical = [field for field in critical_fields if field in extraction_result.missing_fields]
        if len(missing_critical) >= 1:  # Only flag if vessel is missing
            return True
        
        # Don't flag for validation errors unless they're severe
        if extraction_result.data:
            validation = self.validate_extracted_data(extraction_result.data)
            # Only flag if there are multiple severe validation errors
            if not validation['is_valid'] and len(validation.get('errors', [])) > 3:
                return True
        
        return False