#!/usr/bin/env python3
"""
Secure Web UI for Crew Feedback Parser
Requires users to input their OpenAI API key before accessing functionality
"""

import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, session
import openai
from werkzeug.utils import secure_filename
import logging

# Import our modules
from crew_feedback_parser.services.batch_processor import BatchProcessor
from crew_feedback_parser.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a random secret key for sessions

# Configuration
UPLOAD_FOLDER = 'web_uploads'
OUTPUT_FOLDER = 'web_output'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def validate_openai_key(api_key):
    """Validate OpenAI API key by making a test request"""
    try:
        # Create a temporary client with the provided key
        client = openai.OpenAI(api_key=api_key)
        
        # Make a simple test request
        response = client.models.list()
        
        # If we get here, the key is valid
        return True, "API key is valid"
    
    except openai.AuthenticationError:
        return False, "Invalid API key"
    except openai.RateLimitError:
        return False, "API key rate limit exceeded"
    except openai.APIError as e:
        return False, f"OpenAI API error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

@app.route('/')
def index():
    """Serve the secure main page"""
    return render_template('secure_index.html')

@app.route('/validate-api-key', methods=['POST'])
def validate_api_key_endpoint():
    """Validate the provided OpenAI API key"""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({'valid': False, 'error': 'API key is required'})
        
        if not api_key.startswith('sk-'):
            return jsonify({'valid': False, 'error': 'Invalid API key format'})
        
        # Validate the key
        is_valid, message = validate_openai_key(api_key)
        
        if is_valid:
            # Store the API key in session (server-side storage)
            session['openai_api_key'] = api_key
            logger.info("API key validated successfully")
            return jsonify({'valid': True, 'message': message})
        else:
            logger.warning(f"API key validation failed: {message}")
            return jsonify({'valid': False, 'error': message})
    
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        return jsonify({'valid': False, 'error': 'Validation failed'})

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and processing"""
    try:
        # Check if API key is in session
        api_key = session.get('openai_api_key')
        if not api_key:
            # Also check for API key in headers (for additional security)
            api_key = request.headers.get('X-API-Key')
            if not api_key:
                return jsonify({'success': False, 'error': 'API key required. Please refresh and enter your API key.'})
        
        # Validate API key is still working
        is_valid, message = validate_openai_key(api_key)
        if not is_valid:
            # Clear the invalid key from session
            session.pop('openai_api_key', None)
            return jsonify({'success': False, 'error': f'API key is no longer valid: {message}'})
        
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'})
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'success': False, 'error': 'No files selected'})
        
        # Create a unique session folder for this upload
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        session_upload_folder = os.path.join(UPLOAD_FOLDER, session_id)
        session_output_folder = os.path.join(OUTPUT_FOLDER, session_id)
        
        os.makedirs(session_upload_folder, exist_ok=True)
        os.makedirs(session_output_folder, exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                # Check file size
                file.seek(0, 2)  # Seek to end
                file_size = file.tell()
                file.seek(0)  # Reset to beginning
                
                if file_size > MAX_FILE_SIZE:
                    logger.warning(f"File {file.filename} too large: {file_size} bytes")
                    continue
                
                filename = secure_filename(file.filename)
                file_path = os.path.join(session_upload_folder, filename)
                file.save(file_path)
                saved_files.append(file_path)
                logger.info(f"Saved file: {filename}")
        
        if not saved_files:
            return jsonify({'success': False, 'error': 'No valid files to process'})
        
        # Process files using BatchProcessor with the user's API key
        logger.info(f"Processing {len(saved_files)} files with user's API key")
        
        # Temporarily set the API key in environment for the processor
        original_api_key = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = api_key
        
        try:
            # Initialize config manager for the processor
            from crew_feedback_parser.config.config_manager import ConfigManager
            config_manager = ConfigManager()
            
            processor = BatchProcessor(config_manager)
            start_time = datetime.now()
            
            # Process the files individually and collect results
            results = []
            output_excel_file = os.path.join(session_output_folder, f"crew_feedback_results_{session_id}.xlsx")
            
            # Initialize Excel writer
            from crew_feedback_parser.services.excel_writer import ExcelWriter
            excel_writer = ExcelWriter(output_excel_file)
            excel_writer.create_or_load_workbook()
            
            # Process each file and handle multiple feedbacks
            for file_path in saved_files:
                try:
                    # Process the file - this now returns a list of results (one per feedback)
                    file_results = processor._process_single_file_multiple(Path(file_path))
                    
                    # Add all results from this file
                    results.extend(file_results)
                    
                    # Write each result to Excel
                    for individual_result in file_results:
                        excel_writer.append_feedback_and_log(individual_result)
                    
                    logger.info(f"Processed {file_path}: {len(file_results)} feedback(s) extracted")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    # Create error result
                    from crew_feedback_parser.models.feedback_data import ProcessingResult
                    error_result = ProcessingResult(
                        file_name=Path(file_path).name,
                        status="error",
                        error_message=str(e)
                    )
                    results.append(error_result)
                    excel_writer.append_feedback_and_log(error_result)
            
            # Save Excel file
            excel_writer.save_workbook()
            excel_writer.close_workbook()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Generate statistics
            total_files = len(saved_files)
            successful_files = len([r for r in results if r.status == "pass"])
            failed_files = len([r for r in results if r.status == "fail"])
            error_files = len([r for r in results if r.status == "error"])
            
            # Check if Excel file was created
            if os.path.exists(output_excel_file):
                download_url = f'/download/{session_id}/{Path(output_excel_file).name}'
                
                logger.info(f"Processing complete: {successful_files}/{total_files} files successful, {failed_files} failed, {error_files} errors")
                
                return jsonify({
                    'success': True,
                    'message': f'Processed {successful_files} out of {total_files} files successfully',
                    'download_url': download_url,
                    'stats': {
                        'total_files': total_files,
                        'successful': successful_files,
                        'failed': failed_files + error_files,
                        'processing_time': f"{processing_time:.1f}s"
                    }
                })
            else:
                return jsonify({'success': False, 'error': 'No output file generated'})
        
        finally:
            # Restore original API key
            if original_api_key:
                os.environ['OPENAI_API_KEY'] = original_api_key
            else:
                os.environ.pop('OPENAI_API_KEY', None)
            
            # Clean up uploaded files
            try:
                shutil.rmtree(session_upload_folder)
            except Exception as e:
                logger.warning(f"Failed to clean up upload folder: {e}")
    
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'})

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    """Serve download files"""
    try:
        # Security check: ensure the session_id and filename are safe
        session_id = secure_filename(session_id)
        filename = secure_filename(filename)
        
        file_path = os.path.join(OUTPUT_FOLDER, session_id, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Schedule cleanup after file download (delayed)
        def cleanup_session():
            try:
                session_folder = os.path.join(OUTPUT_FOLDER, session_id)
                shutil.rmtree(session_folder)
                logger.info(f"Cleaned up session folder: {session_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up session folder {session_id}: {e}")
        
        # Schedule cleanup in 10 seconds to allow download to complete
        import threading
        threading.Timer(10.0, cleanup_session).start()
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f"crew_feedback_results_{session_id}.xlsx",
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    
    except Exception as e:
        logger.error(f"Error serving download: {str(e)}")
        return jsonify({'error': 'Download failed'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 50MB per file.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚢 Starting Secure Crew Feedback Parser Web UI...")
    print("📝 Features:")
    print("   • Secure API key input and validation")
    print("   • Session-based key storage")
    print("   • File upload and processing")
    print("   • Excel results download")
    print("   • Automatic cleanup")
    print()
    print("🔐 Security:")
    print("   • API keys stored in server sessions only")
    print("   • Keys validated before processing")
    print("   • No keys saved to disk or logs")
    print("   • Secure file handling")
    print()
    print("🌐 Access the application at: http://localhost:5000")
    print("⚠️  Users must provide their own OpenAI API key to use the service")
    print()
    
    # Set Flask configuration
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5001)