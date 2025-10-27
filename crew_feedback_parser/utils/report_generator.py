"""
Processing summary and reporting utilities for crew feedback parser system.
Generates comprehensive reports of processed files, errors, and statistics.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from ..models.feedback_data import ProcessingResult


class ReportGenerator:
    """
    Generates comprehensive processing reports and statistics for crew feedback parsing.
    Provides detailed analysis of processing results, error patterns, and performance metrics.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize report generator.
        
        Args:
            logger: Logger instance (optional, creates default if not provided)
        """
        self.logger = logger or logging.getLogger(__name__)

    def generate_processing_report(self, 
                                 processing_results: List[ProcessingResult],
                                 processing_stats: Dict[str, Any],
                                 error_summary: Dict[str, Any],
                                 output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive processing report with statistics and analysis.
        
        Args:
            processing_results: List of processing results from batch operation
            processing_stats: Processing statistics dictionary
            error_summary: Error summary from error handler
            output_file: Optional path to save report as JSON file
            
        Returns:
            Dict containing comprehensive processing report
        """
        self.logger.info("Generating comprehensive processing report")
        
        # Generate report sections
        report = {
            'report_metadata': self._generate_report_metadata(),
            'processing_summary': self._generate_processing_summary(processing_stats),
            'file_analysis': self._analyze_file_results(processing_results),
            'error_analysis': self._analyze_errors(processing_results, error_summary),
            'performance_metrics': self._calculate_performance_metrics(processing_stats, processing_results),
            'quality_metrics': self._analyze_data_quality(processing_results),
            'recommendations': self._generate_recommendations(processing_results, error_summary),
            'detailed_results': self._format_detailed_results(processing_results)
        }
        
        # Save to file if requested
        if output_file:
            self._save_report_to_file(report, output_file)
        
        self.logger.info(f"Processing report generated with {len(processing_results)} file results")
        return report

    def _generate_report_metadata(self) -> Dict[str, Any]:
        """Generate report metadata information."""
        return {
            'report_generated_at': datetime.now().isoformat(),
            'report_version': '1.0',
            'generator': 'CrewFeedbackParser ReportGenerator'
        }

    def _generate_processing_summary(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level processing summary."""
        total_processed = stats.get('successful', 0) + stats.get('failed', 0) + stats.get('errors', 0)
        success_rate = (stats.get('successful', 0) / total_processed * 100) if total_processed > 0 else 0
        
        return {
            'total_files_found': stats.get('total_files', 0),
            'total_files_processed': total_processed,
            'successful_extractions': stats.get('successful', 0),
            'failed_extractions': stats.get('failed', 0),
            'processing_errors': stats.get('errors', 0),
            'success_rate_percent': round(success_rate, 2),
            'processing_duration_seconds': stats.get('processing_duration', 0),
            'processing_start_time': stats.get('start_time').isoformat() if stats.get('start_time') else None,
            'processing_end_time': stats.get('end_time').isoformat() if stats.get('end_time') else None
        }

    def _analyze_file_results(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Analyze file processing results by status and patterns."""
        if not results:
            return {'total_files': 0, 'status_breakdown': {}, 'file_types': {}}
        
        # Status breakdown
        status_counts = {'pass': 0, 'fail': 0, 'error': 0}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        # File type analysis
        file_types = {}
        for result in results:
            if result.file_name:
                extension = Path(result.file_name).suffix.lower()
                file_types[extension] = file_types.get(extension, 0) + 1
        
        # Processing time analysis
        processing_times = []
        for result in results:
            if hasattr(result, 'processing_duration') and result.processing_duration:
                processing_times.append(result.processing_duration)
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            'total_files': len(results),
            'status_breakdown': status_counts,
            'file_types_processed': file_types,
            'average_processing_time_seconds': round(avg_processing_time, 2),
            'fastest_processing_time': min(processing_times) if processing_times else 0,
            'slowest_processing_time': max(processing_times) if processing_times else 0
        }

    def _analyze_errors(self, 
                       results: List[ProcessingResult], 
                       error_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error patterns and provide detailed error breakdown."""
        if not results:
            return {'total_errors': 0, 'error_patterns': {}, 'common_issues': []}
        
        # Categorize errors by type
        error_categories = {
            'api_errors': [],
            'file_errors': [],
            'extraction_errors': [],
            'validation_errors': [],
            'other_errors': []
        }
        
        # Analyze error messages
        error_message_patterns = {}
        
        for result in results:
            if result.status in ['fail', 'error'] and result.error_message:
                error_msg = result.error_message.lower()
                
                # Categorize error
                if 'api' in error_msg or 'rate limit' in error_msg or 'timeout' in error_msg:
                    error_categories['api_errors'].append(result)
                elif 'file' in error_msg or 'permission' in error_msg or 'not found' in error_msg:
                    error_categories['file_errors'].append(result)
                elif 'extraction' in error_msg or 'pattern' in error_msg or 'confidence' in error_msg:
                    error_categories['extraction_errors'].append(result)
                elif 'validation' in error_msg or 'invalid' in error_msg:
                    error_categories['validation_errors'].append(result)
                else:
                    error_categories['other_errors'].append(result)
                
                # Track error message patterns
                # Extract key words from error messages
                key_words = self._extract_error_keywords(error_msg)
                for word in key_words:
                    error_message_patterns[word] = error_message_patterns.get(word, 0) + 1
        
        # Find most common error patterns
        common_patterns = sorted(error_message_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate common issues list
        common_issues = self._identify_common_issues(error_categories, common_patterns)
        
        return {
            'total_errors': sum(len(errors) for errors in error_categories.values()),
            'error_categories': {
                category: len(errors) for category, errors in error_categories.items()
            },
            'error_message_patterns': dict(common_patterns),
            'common_issues': common_issues,
            'error_summary_from_handler': error_summary
        }

    def _extract_error_keywords(self, error_message: str) -> List[str]:
        """Extract meaningful keywords from error messages."""
        # Common error-related keywords to track
        keywords = [
            'timeout', 'rate limit', 'api', 'permission', 'not found', 'corrupted',
            'invalid', 'extraction', 'confidence', 'validation', 'network', 'connection',
            'file', 'format', 'size', 'encoding', 'pattern', 'missing'
        ]
        
        found_keywords = []
        for keyword in keywords:
            if keyword in error_message:
                found_keywords.append(keyword)
        
        return found_keywords

    def _identify_common_issues(self, 
                               error_categories: Dict[str, List], 
                               common_patterns: List[tuple]) -> List[str]:
        """Identify common issues based on error analysis."""
        issues = []
        
        # Check for high API error rate
        api_errors = len(error_categories.get('api_errors', []))
        total_errors = sum(len(errors) for errors in error_categories.values())
        
        if total_errors > 0:
            if api_errors / total_errors > 0.3:
                issues.append("High rate of API errors - check network connectivity and API key")
            
            if len(error_categories.get('file_errors', [])) / total_errors > 0.2:
                issues.append("Multiple file access issues - check file permissions and paths")
            
            if len(error_categories.get('extraction_errors', [])) / total_errors > 0.4:
                issues.append("High extraction failure rate - documents may have unusual formats")
        
        # Check common error patterns
        for pattern, count in common_patterns:
            if count > 3:  # If pattern appears more than 3 times
                if pattern == 'timeout':
                    issues.append("Frequent timeout errors - consider increasing timeout settings")
                elif pattern == 'rate limit':
                    issues.append("Rate limiting detected - reduce concurrent processing or add delays")
                elif pattern == 'confidence':
                    issues.append("Low confidence extractions - documents may need preprocessing")
        
        return issues

    def _calculate_performance_metrics(self, 
                                     stats: Dict[str, Any], 
                                     results: List[ProcessingResult]) -> Dict[str, Any]:
        """Calculate detailed performance metrics."""
        total_processed = len(results)
        processing_duration = stats.get('processing_duration', 0)
        
        if total_processed == 0 or processing_duration == 0:
            return {
                'files_per_second': 0,
                'average_file_processing_time': 0,
                'throughput_analysis': 'No data available'
            }
        
        files_per_second = total_processed / processing_duration
        avg_file_time = processing_duration / total_processed
        
        # Analyze throughput
        if files_per_second > 2:
            throughput_analysis = "Excellent throughput"
        elif files_per_second > 1:
            throughput_analysis = "Good throughput"
        elif files_per_second > 0.5:
            throughput_analysis = "Moderate throughput"
        else:
            throughput_analysis = "Low throughput - consider optimization"
        
        # Calculate success rate over time (if timestamps available)
        time_based_analysis = self._analyze_processing_over_time(results)
        
        return {
            'files_per_second': round(files_per_second, 2),
            'average_file_processing_time_seconds': round(avg_file_time, 2),
            'total_processing_time_minutes': round(processing_duration / 60, 2),
            'throughput_analysis': throughput_analysis,
            'time_based_performance': time_based_analysis
        }

    def _analyze_processing_over_time(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Analyze how processing performance changed over time."""
        if not results:
            return {'analysis': 'No data available'}
        
        # Group results by time intervals (if timestamps available)
        timestamped_results = [r for r in results if r.processing_timestamp]
        
        if len(timestamped_results) < 2:
            return {'analysis': 'Insufficient timestamp data for time-based analysis'}
        
        # Sort by timestamp
        timestamped_results.sort(key=lambda x: x.processing_timestamp)
        
        # Calculate success rate in first half vs second half
        midpoint = len(timestamped_results) // 2
        first_half = timestamped_results[:midpoint]
        second_half = timestamped_results[midpoint:]
        
        first_half_success = sum(1 for r in first_half if r.status == 'pass')
        second_half_success = sum(1 for r in second_half if r.status == 'pass')
        
        first_half_rate = (first_half_success / len(first_half) * 100) if first_half else 0
        second_half_rate = (second_half_success / len(second_half) * 100) if second_half else 0
        
        trend = "improving" if second_half_rate > first_half_rate else "declining" if second_half_rate < first_half_rate else "stable"
        
        return {
            'first_half_success_rate': round(first_half_rate, 1),
            'second_half_success_rate': round(second_half_rate, 1),
            'performance_trend': trend,
            'analysis': f"Processing performance was {trend} over time"
        }

    def _analyze_data_quality(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Analyze the quality of extracted data."""
        if not results:
            return {'analysis': 'No data available for quality analysis'}
        
        successful_results = [r for r in results if r.status == 'pass' and r.data]
        
        if not successful_results:
            return {'analysis': 'No successful extractions for quality analysis'}
        
        # Analyze completeness of extracted data
        completeness_scores = []
        field_completion_rates = {}
        
        for result in successful_results:
            if result.data:
                # Calculate completeness score
                total_fields = 14  # Total fields in FeedbackData
                completed_fields = 0
                
                # Check each field
                fields_to_check = [
                    ('vessel', result.data.vessel),
                    ('crew_name', result.data.crew_name),
                    ('crew_rank', result.data.crew_rank),
                    ('safer_with_sos', result.data.safer_with_sos),
                    ('fatigue_monitoring_prevention', result.data.fatigue_monitoring_prevention),
                    ('geofence_awareness', result.data.geofence_awareness),
                    ('heat_stress_alerts_change', result.data.heat_stress_alerts_change),
                    ('work_rest_hour_monitoring', result.data.work_rest_hour_monitoring),
                    ('ptw_system_improvement', result.data.ptw_system_improvement),
                    ('paperwork_error_reduction', result.data.paperwork_error_reduction),
                    ('noise_exposure_monitoring', result.data.noise_exposure_monitoring),
                    ('activity_tracking_awareness', result.data.activity_tracking_awareness),
                    ('fall_detection_confidence', result.data.fall_detection_confidence),
                    ('feature_preference', result.data.feature_preference)
                ]
                
                for field_name, field_value in fields_to_check:
                    if field_value is not None and str(field_value).strip():
                        completed_fields += 1
                        field_completion_rates[field_name] = field_completion_rates.get(field_name, 0) + 1
                
                completeness_score = completed_fields / total_fields
                completeness_scores.append(completeness_score)
        
        # Calculate field completion rates as percentages
        total_successful = len(successful_results)
        field_completion_percentages = {
            field: (count / total_successful * 100) for field, count in field_completion_rates.items()
        }
        
        # Overall quality metrics
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
        # Identify problematic fields (low completion rates)
        problematic_fields = [
            field for field, rate in field_completion_percentages.items() if rate < 70
        ]
        
        return {
            'total_successful_extractions': total_successful,
            'average_completeness_score': round(avg_completeness, 3),
            'field_completion_rates': {
                field: round(rate, 1) for field, rate in field_completion_percentages.items()
            },
            'problematic_fields': problematic_fields,
            'quality_assessment': self._assess_overall_quality(avg_completeness, problematic_fields)
        }

    def _assess_overall_quality(self, avg_completeness: float, problematic_fields: List[str]) -> str:
        """Assess overall data quality based on metrics."""
        if avg_completeness > 0.9 and len(problematic_fields) == 0:
            return "Excellent - High completeness with no problematic fields"
        elif avg_completeness > 0.8 and len(problematic_fields) <= 2:
            return "Good - High completeness with minimal issues"
        elif avg_completeness > 0.6 and len(problematic_fields) <= 4:
            return "Fair - Moderate completeness with some field extraction issues"
        else:
            return "Poor - Low completeness or many problematic fields requiring attention"

    def _generate_recommendations(self, 
                                results: List[ProcessingResult], 
                                error_summary: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on processing results."""
        recommendations = []
        
        if not results:
            return ["No processing results available for analysis"]
        
        total_files = len(results)
        successful = sum(1 for r in results if r.status == 'pass')
        failed = sum(1 for r in results if r.status == 'fail')
        errors = sum(1 for r in results if r.status == 'error')
        
        success_rate = (successful / total_files * 100) if total_files > 0 else 0
        
        # Success rate recommendations
        if success_rate < 50:
            recommendations.append("Critical: Very low success rate (<50%). Review document formats and API configuration.")
        elif success_rate < 70:
            recommendations.append("Warning: Moderate success rate (<70%). Consider document preprocessing or pattern improvements.")
        elif success_rate > 90:
            recommendations.append("Excellent: High success rate (>90%). Current configuration is working well.")
        
        # Error-specific recommendations
        if errors > total_files * 0.2:  # More than 20% errors
            recommendations.append("High error rate detected. Check API connectivity, file permissions, and system resources.")
        
        if failed > total_files * 0.3:  # More than 30% failed extractions
            recommendations.append("High extraction failure rate. Consider improving document quality or extraction patterns.")
        
        # Error pattern recommendations
        error_counts = error_summary.get('error_counts_by_type', {})
        for error_type, count in error_counts.items():
            if count > 3:
                if 'api' in error_type:
                    recommendations.append(f"Frequent API errors ({count}). Check network stability and API limits.")
                elif 'file' in error_type:
                    recommendations.append(f"Multiple file errors ({count}). Verify file permissions and formats.")
                elif 'extraction' in error_type:
                    recommendations.append(f"Extraction issues ({count}). Review document formats and extraction logic.")
        
        # Performance recommendations
        if total_files > 0:
            # Add performance-based recommendations here if needed
            pass
        
        if not recommendations:
            recommendations.append("Processing completed successfully with no major issues identified.")
        
        return recommendations

    def _format_detailed_results(self, results: List[ProcessingResult]) -> List[Dict[str, Any]]:
        """Format detailed results for inclusion in report."""
        detailed_results = []
        
        for result in results:
            result_dict = {
                'file_name': result.file_name,
                'status': result.status,
                'processing_timestamp': result.processing_timestamp.isoformat() if result.processing_timestamp else None,
                'error_message': result.error_message,
                'has_data': result.has_data()
            }
            
            # Add data summary if available
            if result.data:
                result_dict['data_summary'] = {
                    'vessel': bool(result.data.vessel),
                    'crew_name': bool(result.data.crew_name),
                    'crew_rank': bool(result.data.crew_rank),
                    'ratings_count': sum(1 for rating in [
                        result.data.safer_with_sos, result.data.fatigue_monitoring_prevention,
                        result.data.geofence_awareness, result.data.heat_stress_alerts_change,
                        result.data.work_rest_hour_monitoring, result.data.ptw_system_improvement,
                        result.data.paperwork_error_reduction, result.data.noise_exposure_monitoring,
                        result.data.activity_tracking_awareness, result.data.fall_detection_confidence
                    ] if rating is not None),
                    'has_feature_preference': bool(result.data.feature_preference)
                }
            
            detailed_results.append(result_dict)
        
        return detailed_results

    def _save_report_to_file(self, report: Dict[str, Any], output_file: str) -> None:
        """Save report to JSON file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Processing report saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save report to {output_file}: {e}")

    def generate_summary_text(self, report: Dict[str, Any]) -> str:
        """
        Generate a human-readable text summary from the report.
        
        Args:
            report: Processing report dictionary
            
        Returns:
            Formatted text summary
        """
        summary_lines = []
        
        # Header
        summary_lines.append("=== CREW FEEDBACK PROCESSING REPORT ===")
        summary_lines.append(f"Generated: {report['report_metadata']['report_generated_at']}")
        summary_lines.append("")
        
        # Processing Summary
        proc_summary = report['processing_summary']
        summary_lines.append("PROCESSING SUMMARY:")
        summary_lines.append(f"  Total files found: {proc_summary['total_files_found']}")
        summary_lines.append(f"  Files processed: {proc_summary['total_files_processed']}")
        summary_lines.append(f"  Successful: {proc_summary['successful_extractions']}")
        summary_lines.append(f"  Failed: {proc_summary['failed_extractions']}")
        summary_lines.append(f"  Errors: {proc_summary['processing_errors']}")
        summary_lines.append(f"  Success rate: {proc_summary['success_rate_percent']}%")
        summary_lines.append(f"  Processing time: {proc_summary['processing_duration_seconds']:.1f} seconds")
        summary_lines.append("")
        
        # Performance Metrics
        perf_metrics = report['performance_metrics']
        summary_lines.append("PERFORMANCE METRICS:")
        summary_lines.append(f"  Throughput: {perf_metrics['files_per_second']} files/second")
        summary_lines.append(f"  Average processing time: {perf_metrics['average_file_processing_time_seconds']:.2f} seconds/file")
        summary_lines.append(f"  Assessment: {perf_metrics['throughput_analysis']}")
        summary_lines.append("")
        
        # Quality Analysis
        quality_metrics = report['quality_metrics']
        if 'average_completeness_score' in quality_metrics:
            summary_lines.append("DATA QUALITY:")
            summary_lines.append(f"  Average completeness: {quality_metrics['average_completeness_score']:.1%}")
            summary_lines.append(f"  Quality assessment: {quality_metrics['quality_assessment']}")
            if quality_metrics['problematic_fields']:
                summary_lines.append(f"  Problematic fields: {', '.join(quality_metrics['problematic_fields'])}")
            summary_lines.append("")
        
        # Error Analysis
        error_analysis = report['error_analysis']
        if error_analysis['total_errors'] > 0:
            summary_lines.append("ERROR ANALYSIS:")
            summary_lines.append(f"  Total errors: {error_analysis['total_errors']}")
            for category, count in error_analysis['error_categories'].items():
                if count > 0:
                    summary_lines.append(f"  {category.replace('_', ' ').title()}: {count}")
            summary_lines.append("")
        
        # Recommendations
        recommendations = report['recommendations']
        if recommendations:
            summary_lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                summary_lines.append(f"  {i}. {rec}")
            summary_lines.append("")
        
        summary_lines.append("=== END REPORT ===")
        
        return "\n".join(summary_lines)