#!/usr/bin/env python3
"""
Test script to use OpenAI for PDF parsing instead of LlamaIndex.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from crew_feedback_parser.config.config_manager import ConfigManager
from crew_feedback_parser.services.openai_client import OpenAIClient
from crew_feedback_parser.services.data_extractor import DataExtractor
from crew_feedback_parser.services.excel_writer import ExcelWriter

def test_openai_parsing():
    """Test OpenAI-based PDF parsing."""
    
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("   You can get an API key from: https://platform.openai.com/api-keys")
        print("   Then run: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        print("🚀 Testing OpenAI-based PDF parsing...")
        
        # Initialize components
        config_manager = ConfigManager()
        openai_client = OpenAIClient(config_manager)
        data_extractor = DataExtractor()
        
        # Test API connection
        print("🔗 Testing OpenAI API connection...")
        if not openai_client.validate_connection():
            print("❌ OpenAI API connection failed")
            return
        print("✅ OpenAI API connection successful")
        
        # Parse the PDF
        pdf_path = "./test_real_api/2nd Officer Nazri.pdf"
        print(f"📄 Parsing PDF: {pdf_path}")
        
        parse_result = openai_client.parse_document(pdf_path)
        print(f"✅ PDF parsed successfully: {len(parse_result['text'])} characters extracted")
        
        # Extract structured data
        print("🔍 Extracting structured data...")
        extraction_result = data_extractor.extract_complete_data(parse_result['text'])
        
        print(f"✅ Data extraction completed:")
        print(f"   Confidence: {extraction_result.confidence_score:.2f}")
        print(f"   Missing fields: {len(extraction_result.missing_fields)}")
        
        # Show extracted data
        if extraction_result.data:
            data = extraction_result.data
            print(f"\n📊 Extracted Data:")
            print(f"   Vessel: {data.vessel}")
            print(f"   Crew Name: {data.crew_name}")
            print(f"   Crew Rank: {data.crew_rank}")
            print(f"   Safer with SOS: {data.safer_with_sos}")
            print(f"   Feature Preference: {data.feature_preference}")
        
        # Save to Excel
        output_file = "./openai_test_results.xlsx"
        print(f"\n💾 Saving to Excel: {output_file}")
        
        excel_writer = ExcelWriter(output_file)
        excel_writer.create_or_load_workbook()
        
        # Create processing result
        from crew_feedback_parser.models.feedback_data import ProcessingResult
        processing_result = ProcessingResult(
            file_name="2nd Officer Nazri.pdf",
            status="pass",
            data=extraction_result.data
        )
        
        excel_writer.append_feedback_and_log(processing_result)
        excel_writer.save_workbook()
        excel_writer.close_workbook()
        
        print(f"✅ Results saved to: {output_file}")
        print("\n🎉 OpenAI parsing test completed successfully!")
        
        # Show sample of extracted text
        print(f"\n📝 Sample extracted text (first 500 chars):")
        print("-" * 50)
        print(parse_result['text'][:500])
        print("...")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_openai_parsing()