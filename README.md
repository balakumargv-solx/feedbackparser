# Crew Feedback Parser System

A batch processing system that uses OpenAI API to parse crew feedback forms and extract structured data to populate Excel spreadsheets.

## 🚀 Quick Start

### Option 1: Secure Web UI (Recommended)
The secure web interface requires users to input their own OpenAI API key for processing.

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r web_requirements.txt
   ```

2. **Launch Web UI**
   ```bash
   python run_secure_ui.py
   ```

3. **Access Application**
   - Open http://localhost:5000 in your browser
   - Enter your OpenAI API key when prompted
   - Upload your feedback forms and download Excel results

### Option 2: Command Line Interface
For direct command-line usage with pre-configured API key.

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your OpenAI API key
   ```

3. **Run the System**
   ```bash
   python main.py /path/to/feedback/forms output.xlsx
   ```

## 🔐 Security Features

- **No Hardcoded API Keys**: Users provide their own OpenAI API keys
- **Session-Based Storage**: API keys stored securely in server sessions only
- **Key Validation**: API keys are validated before processing
- **Automatic Cleanup**: Temporary files are automatically removed
- **Secure File Handling**: All uploads are validated and sanitized

## Project Structure

```
crew_feedback_parser/
├── config/          # Configuration management
├── models/          # Data models and structures
├── services/        # Core business logic services
└── utils/           # Utility functions
```

## 📋 Requirements

- Python 3.8+
- OpenAI API key (users provide their own)
- Input files: PDF, PNG, JPG, TIFF formats
- Maximum file size: 50MB per file

## 🌐 Web UI Features

- **Drag & Drop Upload**: Easy file selection with drag-and-drop support
- **Real-time Processing**: Live progress updates during processing
- **Batch Processing**: Handle multiple files simultaneously
- **Excel Export**: Download structured results as Excel files
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Available Interfaces

1. **Secure Web UI** (`secure_web_ui.py`) - Recommended for most users
2. **Simple Web UI** (`web_ui.py`) - Basic web interface
3. **Advanced Web UI** (`advanced_web_ui.py`) - Feature-rich interface
4. **Command Line** (`main.py`) - Direct CLI processing

## 📁 Project Structure

```
crew_feedback_parser/
├── config/          # Configuration management
├── models/          # Data models and structures
├── services/        # Core business logic services
├── utils/           # Utility functions
├── templates/       # Web UI templates
├── web_uploads/     # Temporary upload storage
└── web_output/      # Processed results
```

## 🔧 Environment Variables

See `.env.example` for all available configuration options. Note that the secure web UI doesn't require pre-configured API keys.