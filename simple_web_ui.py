#!/usr/bin/env python3
"""
Simple Web UI for Crew Feedback Parser System.
Minimal Flask application with file upload and Excel download.
"""

import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, request, send_file, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'crew_feedback_parser_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'web_output'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Simple HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Crew Feedback Parser</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .container { background: #f9f9f9; padding: 30px; border-radius: 10px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .upload-area:hover { border-color: #007bff; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .btn:disabled { background: #ccc; cursor: not-allowed; }
        .status { margin: 20px 0; padding: 15px; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .file-list { margin: 20px 0; }
        .file-item { background: white; padding: 10px; margin: 5px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚢 Crew Feedback Parser</h1>
        <p>Upload your crew feedback forms and get structured Excel results</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>📁 Click here to select files</p>
                <p><small>Supported: PDF, PNG, JPG, JPEG, TIFF (Max 50MB each)</small></p>
                <input type="file" id="fileInput" name="files" multiple accept=".pdf,.png,.jpg,.jpeg,.tiff" style="display: none;">
            </div>
            
            <div id="fileList" class="file-list" style="display: none;"></div>
            
            <button type="button" id="uploadBtn" class="btn" onclick="uploadFiles()" disabled>
                Select Files First
            </button>
            <button type="button" class="btn" onclick="clearFiles()">Clear</button>
        </form>
        
        <div id="status" class="status" style="display: none;"></div>
        <div id="downloadSection" style="display: none; text-align: center; margin: 20px 0;">
            <h3>✅ Processing Complete!</h3>
            <button id="downloadBtn" class="btn">📊 Download Excel Results</button>
        </div>
    </div>

    <script>
        let selectedFiles = [];
        let downloadUrl = null;

        document.getElementById('fileInput').addEventListener('change', function(e) {
            selectedFiles = Array.from(e.target.files);
            updateFileList();
            updateUploadButton();
        });

        function updateFileList() {
            const fileList = document.getElementById('fileList');
            if (selectedFiles.length === 0) {
                fileList.style.display = 'none';
                return;
            }

            fileList.style.display = 'block';
            fileList.innerHTML = '<h3>Selected Files:</h3>';
            selectedFiles.forEach(file => {
                const div = document.createElement('div');
                div.className = 'file-item';
                div.innerHTML = `${file.name} (${formatFileSize(file.size)})`;
                fileList.appendChild(div);
            });
        }

        function updateUploadButton() {
            const btn = document.getElementById('uploadBtn');
            if (selectedFiles.length === 0) {
                btn.disabled = true;
                btn.textContent = 'Select Files First';
            } else {
                btn.disabled = false;
                btn.textContent = `Process ${selectedFiles.length} File${selectedFiles.length > 1 ? 's' : ''}`;
            }
        }

        function clearFiles() {
            selectedFiles = [];
            document.getElementById('fileInput').value = '';
            updateFileList();
            updateUploadButton();
            hideStatus();
            hideDownload();
        }

        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        function uploadFiles() {
            if (selectedFiles.length === 0) return;

            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });

            showStatus('Processing files...', 'info');
            document.getElementById('uploadBtn').disabled = true;
            document.getElementById('uploadBtn').textContent = 'Processing...';

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    showStatus(`Success! Processed ${data.stats.total_files} files. Success: ${data.stats.successful}, Failed: ${data.stats.failed}`, 'success');
                    showDownload(data.download_url);
                } else {
                    showStatus('Error: ' + (data.error || 'Processing failed'), 'error');
                }
            })
            .catch(error => {
                console.error('Upload error:', error);
                showStatus('Upload failed: ' + error.message, 'error');
            })
            .finally(() => {
                document.getElementById('uploadBtn').disabled = false;
                updateUploadButton();
            });
        }

        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${type}`;
            status.style.display = 'block';
        }

        function hideStatus() {
            document.getElementById('status').style.display = 'none';
        }

        function showDownload(url) {
            downloadUrl = url;
            document.getElementById('downloadBtn').onclick = () => window.location.href = url;
            document.getElementById('downloadSection').style.display = 'block';
        }

        function hideDownload() {
            document.getElementById('downloadSection').style.display = 'none';
        }

        // Initialize
        updateUploadButton();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Main upload page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and processing."""
    print("Upload request received")
    
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
                
                # Check file size (50MB limit)
                if os.path.getsize(file_path) > 50 * 1024 * 1024:
                    os.remove(file_path)
                    return jsonify({'error': f'File {filename} is too large (>50MB)'}), 400
                
                uploaded_files.append(filename)
                print(f"Saved file: {filename}")
            else:
                return jsonify({'error': f'Invalid file type: {file.filename}'}), 400
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        print(f"Processing {len(uploaded_files)} files...")
        
        # Import and process files
        try:
            from crew_feedback_parser.config.config_manager import ConfigManager
            from crew_feedback_parser.services.batch_processor import BatchProcessor
            
            config_manager = ConfigManager()
            batch_processor = BatchProcessor(config_manager)
            
            # Validate configuration (skip API check)
            if not batch_processor.validate_configuration(skip_api_check=True):
                return jsonify({'error': 'System configuration error'}), 500
            
            # Output file
            output_filename = f'crew_feedback_results_{session_id}.xlsx'
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            print(f"Processing to: {output_path}")
            
            # Process the uploaded files
            processing_report = batch_processor.process_directory(
                input_directory=session_folder,
                output_excel_file=output_path,
                max_workers=1  # Single worker for web UI stability
            )
            
            # Clean up uploaded files
            shutil.rmtree(session_folder)
            print("Cleaned up temporary files")
            
            # Extract stats safely from processing report
            stats = {
                'total_files': len(uploaded_files),
                'successful': 0,
                'failed': 0,
                'errors': 0
            }
            
            # Try to get stats from processing report if available
            try:
                if 'processing_summary' in processing_report:
                    summary = processing_report['processing_summary']
                    stats.update({
                        'total_files': summary.get('total_files', len(uploaded_files)),
                        'successful': summary.get('successful_extractions', 0),
                        'failed': summary.get('failed_extractions', 0),
                        'errors': summary.get('processing_errors', 0)
                    })
                elif 'summary' in processing_report:
                    summary = processing_report['summary']
                    stats.update({
                        'total_files': summary.get('total_files', len(uploaded_files)),
                        'successful': summary.get('successful', 0),
                        'failed': summary.get('failed', 0),
                        'errors': summary.get('errors', 0)
                    })
            except Exception as e:
                print(f"Warning: Could not extract detailed stats: {e}")
            
            return jsonify({
                'success': True,
                'message': f'Successfully processed {len(uploaded_files)} files',
                'download_url': f'/download/{output_filename}',
                'stats': stats
            })
            
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed Excel file."""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({'error': 'Download failed'}), 500

@app.route('/status')
def status():
    """Check system status."""
    return jsonify({
        'status': 'healthy',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': 50
    })

if __name__ == '__main__':
    print("🚀 Starting Simple Crew Feedback Parser Web UI...")
    print("📁 Upload folder:", os.path.abspath(UPLOAD_FOLDER))
    print("📊 Output folder:", os.path.abspath(OUTPUT_FOLDER))
    print("🌐 Open your browser to: http://localhost:5000")
    print("📋 Supported formats:", ', '.join(ALLOWED_EXTENSIONS))
    print("📏 Max file size: 50MB")
    print("\n✨ Ready to process crew feedback forms!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)