#!/usr/bin/env python3
"""
Web UI for Crew Feedback Parser System.
Simple Flask application with drag-and-drop file upload and Excel download.
"""

import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, send_file, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import logging

from crew_feedback_parser.config.config_manager import ConfigManager, ConfigurationError
from crew_feedback_parser.services.batch_processor import BatchProcessor
from crew_feedback_parser.utils.logging_config import setup_logging

app = Flask(__name__)
app.secret_key = 'crew_feedback_parser_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'web_output'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize logging
logging_config, error_handler = setup_logging(log_level="INFO", enable_console=False)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size_mb(file_path):
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)

@app.route('/')
def index():
    """Main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and processing."""
    logger.info("Upload request received")
    try:
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Create unique session folder
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        uploaded_files = []
        
        # Save uploaded files
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(session_folder, filename)
                file.save(file_path)
                
                # Check file size
                if get_file_size_mb(file_path) > 50:
                    os.remove(file_path)
                    return jsonify({'error': f'File {filename} is too large (>50MB)'}), 400
                
                uploaded_files.append(filename)
            else:
                return jsonify({'error': f'Invalid file type: {file.filename}'}), 400
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        # Process files
        try:
            config_manager = ConfigManager()
            batch_processor = BatchProcessor(config_manager)
            
            # Validate configuration (skip API check for web UI)
            if not batch_processor.validate_configuration(skip_api_check=True):
                return jsonify({'error': 'System configuration error. Please check logs.'}), 500
            
            # Output file
            output_filename = f'crew_feedback_results_{session_id}.xlsx'
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Process the uploaded files
            processing_report = batch_processor.process_directory(
                input_directory=session_folder,
                output_excel_file=output_path,
                max_workers=2  # Limit workers for web UI
            )
            
            # Clean up uploaded files
            import shutil
            shutil.rmtree(session_folder)
            
            return jsonify({
                'success': True,
                'message': f'Successfully processed {len(uploaded_files)} files',
                'download_url': f'/download/{output_filename}',
                'stats': {
                    'total_files': processing_report['processing_summary']['total_files'],
                    'successful': processing_report['processing_summary']['successful_extractions'],
                    'failed': processing_report['processing_summary']['failed_extractions'],
                    'errors': processing_report['processing_summary']['processing_errors'],
                    'processing_time': f"{processing_report['processing_summary']['processing_duration_seconds']:.1f}s"
                }
            })
            
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            return jsonify({'error': f'Configuration error: {e}'}), 500
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed Excel file."""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            flash('File not found or has expired', 'error')
            return redirect(url_for('index'))
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        logger.error(f"Download error: {e}")
        flash('Error downloading file', 'error')
        return redirect(url_for('index'))

@app.route('/status')
def status():
    """Check system status."""
    try:
        config_manager = ConfigManager()
        batch_processor = BatchProcessor(config_manager)
        
        # Check configuration (skip API check for web UI)
        config_valid = batch_processor.validate_configuration(skip_api_check=True)
        
        return jsonify({
            'status': 'healthy' if config_valid else 'configuration_error',
            'configuration_valid': config_valid,
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'max_file_size_mb': 50
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/cleanup')
def cleanup_old_files():
    """Clean up old files (older than 1 hour)."""
    try:
        import time
        current_time = time.time()
        cleaned_count = 0
        
        # Clean upload folders
        for item in os.listdir(UPLOAD_FOLDER):
            item_path = os.path.join(UPLOAD_FOLDER, item)
            if os.path.isdir(item_path):
                if current_time - os.path.getctime(item_path) > 3600:  # 1 hour
                    import shutil
                    shutil.rmtree(item_path)
                    cleaned_count += 1
        
        # Clean output files
        for item in os.listdir(OUTPUT_FOLDER):
            item_path = os.path.join(OUTPUT_FOLDER, item)
            if os.path.isfile(item_path):
                if current_time - os.path.getctime(item_path) > 3600:  # 1 hour
                    os.remove(item_path)
                    cleaned_count += 1
        
        return jsonify({
            'success': True,
            'cleaned_files': cleaned_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Crew Feedback Parser Web UI...")
    print("📁 Upload folder:", os.path.abspath(UPLOAD_FOLDER))
    print("📊 Output folder:", os.path.abspath(OUTPUT_FOLDER))
    print("🌐 Open your browser to: http://localhost:5000")
    print("📋 Supported formats:", ', '.join(ALLOWED_EXTENSIONS))
    print("📏 Max file size: 50MB")
    print("\n✨ Ready to process crew feedback forms!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)