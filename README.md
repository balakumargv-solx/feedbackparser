# Crew Feedback Parser System

A batch processing system that uses LlamaIndex API to parse crew feedback forms and extract structured data to populate Excel spreadsheets.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your LlamaIndex API key
   ```

3. **Run the System**
   ```bash
   python main.py /path/to/feedback/forms output.xlsx
   ```

## Project Structure

```
crew_feedback_parser/
├── config/          # Configuration management
├── models/          # Data models and structures
├── services/        # Core business logic services
└── utils/           # Utility functions
```

## Requirements

- Python 3.8+
- LlamaIndex API key
- Input files: PDF, PNG, JPG, TIFF formats

## Environment Variables

See `.env.example` for all available configuration options.