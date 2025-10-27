#!/usr/bin/env python3
"""
Advanced Web UI for Crew Feedback Parser System.
Professional Flask application with real-time progress tracking and bulk processing dashboard.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, request, send_file, jsonify
from werkzeug.utils import secure_filename
from threading import Thread
import time

app = Flask(__name__)
app.secret_key = 'crew_feedback_parser_advanced_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'web_output'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global processing status storage
processing_sessions = {}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS# 
Advanced HTML template with real-time progress
ADVANCED_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Crew Feedback Parser - Advanced Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .dashboard {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .main-content {
            padding: 30px;
        }
        
        .upload-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: white;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 20px;
        }
        
        .upload-area:hover { border-color: #667eea; background: #f0f4ff; }
        .upload-area.dragover { border-color: #667eea; background: #e8f2ff; }
        
        .file-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .file-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            transition: all 0.3s ease;
        }
        
        .file-card:hover { box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        
        .file-name {
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
            word-break: break-word;
        }
        
        .file-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9em;
            color: #666;
        }
        
        .file-status {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
        }
        
        .status-pending { background: #fff3cd; color: #856404; }
        .status-processing { background: #cce5ff; color: #004085; }
        .status-success { background: #d4edda; color: #155724; }
        .status-failed { background: #f8d7da; color: #721c24; }
        .status-error { background: #f8d7da; color: #721c24; }
        
        .progress-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            display: none;
        }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .progress-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        
        .progress-bar-container {
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.5s ease;
        }
        
        .progress-text {
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 5px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        }
        
        .results-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            display: none;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .result-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert {
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .alert-info { background: #cce5ff; color: #004085; border: 1px solid #b3d7ff; }
        
        .download-section {
            text-align: center;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 15px;
            margin: 20px 0;
            display: none;
        }
        
        .download-btn {
            font-size: 1.2em;
            padding: 15px 30px;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🚢 Crew Feedback Parser Dashboard</h1>
            <p>Advanced bulk processing with real-time progress tracking</p>
        </div>

        <div class="main-content">
            <!-- Upload Section -->
            <div class="upload-section">
                <h2>📁 File Upload</h2>
                <div class="upload-area" id="uploadArea">
                    <div style="font-size: 3em; margin-bottom: 15px;">📄</div>
                    <div style="font-size: 1.2em; margin-bottom: 10px;">Drag & Drop PDF files here</div>
                    <div style="color: #666;">or click to browse files</div>
                    <input type="file" id="fileInput" multiple accept=".pdf,.png,.jpg,.jpeg,.tiff" style="display: none;">
                </div>
                
                <div id="fileGrid" class="file-grid" style="display: none;"></div>
                
                <div style="text-align: center; margin-top: 20px;">
                    <button class="btn" id="processBtn" onclick="startProcessing()" disabled>
                        Select Files First
                    </button>
                    <button class="btn btn-danger" onclick="clearFiles()">Clear All</button>
                </div>
            </div>

            <!-- Progress Section -->
            <div class="progress-section" id="progressSection">
                <div class="progress-header">
                    <h2>⚡ Processing Progress</h2>
                    <div id="processingStatus">Initializing...</div>
                </div>
                
                <div class="progress-stats">
                    <div class="stat-card">
                        <div class="stat-number" id="totalFiles">0</div>
                        <div class="stat-label">Total Files</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="processedFiles">0</div>
                        <div class="stat-label">Processed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="successfulFiles">0</div>
                        <div class="stat-label">Successful</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="failedFiles">0</div>
                        <div class="stat-label">Failed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="processingTime">0s</div>
                        <div class="stat-label">Time Elapsed</div>
                    </div>
                </div>
                
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progressBar"></div>
                </div>
                <div class="progress-text" id="progressText">Ready to start...</div>
            </div>

            <!-- Results Section -->
            <div class="results-section" id="resultsSection">
                <h2>📊 Processing Results</h2>
                <div id="resultsGrid" class="results-grid"></div>
            </div>

            <!-- Download Section -->
            <div class="download-section" id="downloadSection">
                <h2>✅ Processing Complete!</h2>
                <p>Your Excel file with all results is ready</p>
                <button class="btn btn-success download-btn" id="downloadBtn">
                    📊 Download Complete Results
                </button>
                <div style="margin-top: 15px;">
                    <button class="btn" onclick="startNewSession()">Process More Files</button>
                </div>
            </div>

            <!-- Status Messages -->
            <div id="alertContainer"></div>
        </div>
    </div>

    <script>
        let selectedFiles = [];
        let currentSession = null;
        let progressInterval = null;
        let startTime = null;

        // Initialize
        document.getElementById('uploadArea').addEventListener('click', () => 
            document.getElementById('fileInput').click()
        );
        
        document.getElementById('fileInput').addEventListener('change', handleFileSelect);
        
        // Drag and drop handlers
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = Array.from(e.dataTransfer.files);
            addFiles(files);
        });

        function handleFileSelect(e) {
            const files = Array.from(e.target.files);
            addFiles(files);
        }

        function addFiles(files) {
            const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/tiff'];
            const maxSize = 50 * 1024 * 1024; // 50MB

            files.forEach(file => {
                if (!allowedTypes.includes(file.type)) {
                    showAlert(`File ${file.name} is not supported`, 'error');
                    return;
                }

                if (file.size > maxSize) {
                    showAlert(`File ${file.name} is too large (max 50MB)`, 'error');
                    return;
                }

                if (selectedFiles.find(f => f.name === file.name)) {
                    showAlert(`File ${file.name} already selected`, 'error');
                    return;
                }

                selectedFiles.push({
                    file: file,
                    name: file.name,
                    size: file.size,
                    status: 'pending'
                });
            });

            updateFileGrid();
            updateProcessButton();
        }

        function updateFileGrid() {
            const grid = document.getElementById('fileGrid');
            
            if (selectedFiles.length === 0) {
                grid.style.display = 'none';
                return;
            }

            grid.style.display = 'grid';
            grid.innerHTML = '';

            selectedFiles.forEach((fileObj, index) => {
                const card = document.createElement('div');
                card.className = 'file-card';
                card.innerHTML = `
                    <div class="file-name">${fileObj.name}</div>
                    <div class="file-info">
                        <span>${formatFileSize(fileObj.size)}</span>
                        <span class="file-status status-${fileObj.status}">${fileObj.status}</span>
                    </div>
                    <button onclick="removeFile(${index})" style="margin-top: 10px; padding: 5px 10px; border: none; background: #e74c3c; color: white; border-radius: 5px; cursor: pointer;">Remove</button>
                `;
                grid.appendChild(card);
            });
        }

        function removeFile(index) {
            selectedFiles.splice(index, 1);
            updateFileGrid();
            updateProcessButton();
        }

        function clearFiles() {
            selectedFiles = [];
            document.getElementById('fileInput').value = '';
            updateFileGrid();
            updateProcessButton();
            hideAllSections();
        }

        function updateProcessButton() {
            const btn = document.getElementById('processBtn');
            if (selectedFiles.length === 0) {
                btn.disabled = true;
                btn.textContent = 'Select Files First';
            } else {
                btn.disabled = false;
                btn.textContent = `Process ${selectedFiles.length} File${selectedFiles.length > 1 ? 's' : ''}`;
            }
        }

        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }