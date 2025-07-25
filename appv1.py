import streamlit as st
import pdfplumber
import openai
import json
import io
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
import os   
from typing import Dict, Any, Optional, List
import logging
import re
from datetime import datetime, date
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFParser:
    """Module for parsing PDFs and extracting text from both text-based and image-based PDFs"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """
        Extract text from PDF using pdfplumber and OCR for images
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            str: Extracted text content
        """
        text_content = ""
        
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text directly from PDF
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        text_content += f"\n--- Page {page_num + 1} ---\n"
                        text_content += page_text
                    else:
                        # If no text found, try OCR on images
                        logger.info(f"No text found on page {page_num + 1}, attempting OCR...")
                        
                        # Convert page to image for OCR
                        try:
                            # Get page as image
                            page_image = page.within_bbox(page.bbox).to_image(resolution=300)
                            
                            # Convert to PIL Image
                            pil_image = page_image.original
                            
                            # Perform OCR
                            ocr_text = pytesseract.image_to_string(pil_image)
                            
                            if ocr_text.strip():
                                text_content += f"\n--- Page {page_num + 1} (OCR) ---\n"
                                text_content += ocr_text
                            
                        except Exception as ocr_error:
                            logger.warning(f"OCR failed for page {page_num + 1}: {str(ocr_error)}")
                            text_content += f"\n--- Page {page_num + 1} (OCR Failed) ---\n"
                            text_content += "[Could not extract text from this page]\n"
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to parse PDF: {str(e)}")
        
        return text_content.strip()

import json

class JobExtractor:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

    def extract_jobs_from_description(self, full_text: str) -> List[Dict[str, str]]:
        system_message = (
            "You are an AI assistant that extracts job positions from job advertisements. "
            "For each job found, return a valid JSON array. Each job should have: "
            "`title`, `description`, `section_start`, and `section_end` (line numbers)."
        )

        user_message = f"""
Extract all job positions from this job advertisement.

Return a **valid JSON array**, for example:

[
  {{
    "title": "Job Title",
    "description": "Full job text here...",
    "section_start": 12,
    "section_end": 25
  }},
  ...
]

Only return the JSON. No extra text or explanation.

Job Advertisement:
{full_text}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2
            )

            raw_output = response.choices[0].message.content.strip()
            jobs = json.loads(raw_output)  # 🔥 You must catch exceptions here

            return jobs

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            raise RuntimeError("Job extraction failed: invalid JSON returned by GPT")

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"Job extraction failed: {e}")




class PromptGenerator:
    """Module for generating prompts to send to GPT-4"""
    
    @staticmethod
    def create_evaluation_prompt(job_description: str, resume: str, qualification_cert: str = "") -> str:
        """
        Create a structured prompt for GPT-4 to evaluate candidate qualification
        
        Args:
            job_description: Extracted job description text for specific position
            resume: Extracted resume text
            qualification_cert: Extracted qualification certificate text
            
        Returns:
            str: Formatted prompt for GPT-4
        """
        
        current_date = date.today().strftime("%Y-%m-%d")
        cert_section = ""
        if qualification_cert.strip():
            cert_section = f"""
**QUALIFICATION CERTIFICATE:**
{qualification_cert}
"""
        
        prompt = f"""
You are an expert HR evaluator. Your task is to analyze a specific job section with main focus on minimum requirements specified in the job advert  and a candidate's documents to determine if the candidate qualifies for the position.

**EVALUATION CRITERIA:**
1. **Years of Experience**: Check if the candidate meets the minimum years of experience specified in the job description. 
   - Post-qualifying experience should be counted from the date of graduation/certification award ONLY
   - Look for graduation dates, certificate award dates, or degree completion dates
   - Calculate years from the certification/graduation date to {current_date}

2. **Education/Degree**: Verify if the candidate has the required degree or related field as specified in the job description
   - Check the qualification certificate for educational credentials
   - Consider related fields and equivalent qualifications

3. **Skills Match**: Assess if the candidate's skills align with job requirements

**SPECIFIC JOB SECTION FOR EVALUATION:**
{job_description}

**CANDIDATE RESUME:**
{resume}
{cert_section}
**IMPORTANT INSTRUCTIONS:**
- Focus ONLY on the requirements mentioned in the specific job section provided above
- Be objective in your assessment based on the specific job requirements=
- Extract specific requirements (years of experience, education level, skills) from the job section
- Provide a clear reason for your decision including specific dates if found

**RESPONSE FORMAT:**
You must respond with a valid JSON object in exactly this format:
{{
    "qualifies": "Yes" or "No",
    "reason": "Provide a clear explanation of why the candidate does or does not qualify. Include specific information about: 1) Graduation/certification date found (if any), 2) Post-qualifying experience calculated in years, 3) Whether minimum experience requirement is met, 4) Educational qualification assessment, 5) Skills match assessment, 6) Overall recommendation based on the specific job requirements.",
    "graduation_date_found": "Date found in documents or 'Not speficied",
    "post_qualifying_experience_years": "Number of years calculated in years or 'Unable to calculate'",
    "minimum_experience_required": "Years required as stated in job description or 'Not specified'",
    "education_requirement_met": "Yes/No - based on degree requirements in job section",
    "key_requirements_from_job": "List 3-5 key requirements found in the job section"
}}

Do not include any text outside the JSON object.
"""
        return prompt

class GPTEvaluator:
    """Module for interacting with GPT-4 API"""
    
    def __init__(self, api_key: str):
        """
        Initialize GPT-4 evaluator
        
        Args:
            api_key: OpenAI API key
        """
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
    
    def evaluate_candidate(self, prompt: str) -> Dict[str, Any]:
        """
        Send prompt to GPT-4 and get evaluation result
        
        Args:
            prompt: Formatted prompt for evaluation
            
        Returns:
            Dict containing qualification decision and reason
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert HR evaluator. Always respond with valid JSON only. Focus on post-qualifying experience calculation from graduation/certification dates to determine if a candidate meets minimum experience. A candidate qualifies if they meet the minimum years of experience and education requirements specified in the job section provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Extract and parse JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Clean response text (remove any markdown formatting)
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate required fields
            required_fields = ["qualifies", "reason"]
            if not all(field in result for field in required_fields):
                raise ValueError("Invalid response format from GPT-4")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT-4 JSON response: {e}")
            return {
                "qualifies": "No",
                "reason": f"Error processing evaluation: Invalid response format",
                "graduation_date_found": "Error",
                "post_qualifying_experience_years": "Error",
                "minimum_experience_required": "Error",
                "education_requirement_met": "Error",
                "key_requirements_from_job": "Error processing requirements"
            }
        except Exception as e:
            logger.error(f"GPT-4 API error: {e}")
            return {
                "qualifies": "No",
                "reason": f"Error processing evaluation: {str(e)}",
                "graduation_date_found": "Error",
                "post_qualifying_experience_years": "Error",
                "minimum_experience_required": "Error",
                "education_requirement_met": "Error",
                "key_requirements_from_job": "Error processing requirements"
            }

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="ZRA Hiring Automation PoC",
        page_icon="👥",
        layout="centered"  
    )
    
    # Logo Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Logo placeholder - replace with your actual logo
        st.image("image.png")
    
    # Header
    st.title(":blue[ZRA-AI RECRUITMENT AUTOMATION]", width="content")
    st.markdown("##### AI-Powered Candidate Evaluation for Multiple Job Positions")
    
    # Initialize session state
    if 'jobs_extracted' not in st.session_state:
        st.session_state.jobs_extracted = False
    if 'available_jobs' not in st.session_state:
        st.session_state.available_jobs = []
    if 'selected_job' not in st.session_state:
        st.session_state.selected_job = None
    
    # API Key Configuration in main area (not sidebar)
    st.markdown("---")
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Enter your OpenAI API key here"
    )
    
    # Main content area - Single column layout
    if api_key:
        st.markdown("---")
        
        # Step 1: Upload Job Description
        st.markdown("#### Job Advert")
        job_file = st.file_uploader(
            "Upload Job Advertisement (PDF) - Can contain multiple positions",
            type="pdf",
            key="job_description"
        )
        
        if job_file and not st.session_state.jobs_extracted:
            with st.spinner("Extracting job positions"):
                try:
                    pdf_parser = PDFParser()
                    job_extractor = JobExtractor(api_key)  # Initialize with API key
                    
                   
                    job_text = pdf_parser.extract_text_from_pdf(job_file)
                    
             
                    available_jobs = job_extractor.extract_jobs_from_description(job_text)
                    
                    st.session_state.available_jobs = available_jobs
                    st.session_state.jobs_extracted = True
                    st.session_state.full_job_text = job_text
                    
                    st.success(f"Identified {len(available_jobs)} job position(s)")
                    
                except Exception as e:
                    st.error(f"Error processing job advertisement: {str(e)}")
                    logger.error(f"Job extraction error: {e}")
        
        # Step 2: Select Job Position
        if st.session_state.jobs_extracted and st.session_state.available_jobs:
            st.markdown("---")
            st.markdown("#### Select Job Position")
            
            # Display found jobs count
            st.info(f"Found {len(st.session_state.available_jobs)} job position(s) in the advertisement")
            
            job_options = [job['title'] for job in st.session_state.available_jobs]
            selected_job_title = st.selectbox(
                "Select the job position you want to shortlist for:",
                options=job_options,
                index=0,
                help="Select the specific job position from the extracted list"
            )
            
            # Find selected job details
            selected_job = next(job for job in st.session_state.available_jobs if job['title'] == selected_job_title)
            st.session_state.selected_job = selected_job
            
            # Show selected job description in a more organized way
            st.success(f"Selected Position: **{selected_job_title}**")
            
            # Show job section details
            with st.expander(f"View Complete Job Section: {selected_job_title}"):
                st.markdown("**Job Title:**")
                st.write(f"*{selected_job_title}*")
                st.markdown("**Job Section Content:**")
                st.text_area(
                    "Job Description", 
                    selected_job['description'], 
                    height=300,
                    key=f"job_desc_{selected_job_title}"
                )
                
                # Show section information if available
                if 'section_start' in selected_job and 'section_end' in selected_job:
                    st.caption(f"Section extracted from lines {selected_job['section_start']} to {selected_job['section_end']} of the document")
            
            # Show all available jobs for reference
            with st.expander("View All Available Positions"):
                for i, job in enumerate(st.session_state.available_jobs, 1):
                    st.markdown(f"**{i}. {job['title']}**")
                    if selected_job_title == job['title']:
                        st.markdown("   ✅ *Currently Selected*")
            
            # Step 3: Upload Candidate Documents
            st.markdown("---")
            st.markdown("#### Candidate Documents")
            
            # Resume and certificate upload in single column
            st.markdown("###### Curriculum Vitae (CV)")
            resume_file = st.file_uploader(
                "Upload Candidate CV (PDF)",
                type="pdf",
                key="resume"
            )
            
            if resume_file:
                st.success(f"Curriculum Vitae (CV) uploaded: {resume_file.name}")
            
            st.markdown("###### Qualification Certificate")
            cert_file = st.file_uploader(
                "Upload Qualification Certificate (PDF)",
                type="pdf",
                key="certificate",
                help="Upload degree certificate, diploma, or professional certification"
            )
            
            if cert_file:
                st.success(f"Certificate uploaded: {cert_file.name}")
            
            # Step 4: Evaluation
            if resume_file:  # Certificate is optional
                st.markdown("---")
                st.markdown("#### Candidate Evaluation")
                
                if st.button("Evaluate Candidate", type="primary", use_container_width=True):
                    
                    with st.spinner("Processing documents and evaluating candidate..."):
                        try:
                            # Initialize components
                            pdf_parser = PDFParser()
                            prompt_generator = PromptGenerator()
                            gpt_evaluator = GPTEvaluator(api_key)
                            
                        
                            resume_text = pdf_parser.extract_text_from_pdf(resume_file)
                            
                            # Extract text from certificate (if provided)
                            cert_text = ""
                            if cert_file:
                                cert_text = pdf_parser.extract_text_from_pdf(cert_file)
                            
                            # Generate prompt using selected job description
                    
                            prompt = prompt_generator.create_evaluation_prompt(
                                selected_job['description'], 
                                resume_text, 
                                cert_text
                            )
                            
                            # Get GPT-4 evaluation
                        
                            evaluation_result = gpt_evaluator.evaluate_candidate(prompt)
                            
                            # Display results
                            st.markdown("---")
                            st.markdown("#### Evaluation Results")
                            
                            # Create result display
                            if evaluation_result["qualifies"].lower() == "yes":
                                st.success("✅ **QUALIFIED**")
                        
                            else:
                                st.error("❌ **NOT QUALIFIED**")
                            
                            st.markdown(f"**Position:** {selected_job_title}")
                            
                            # Display detailed evaluation
                            st.markdown("#### Detailed Evaluation")
                            st.write(evaluation_result["reason"])
                            
                            # Display extracted metrics if available
                            if "graduation_date_found" in evaluation_result:
                                col1, col2, col3 = st.columns([3, 3, 3])
                                with col1:
                                    st.metric(
                                        "Graduation Date", 
                                        evaluation_result.get("graduation_date_found", "Not found")
                                    )
                                with col2:
                                    st.metric(
                                        "Post-Qualifying Experience", 
                                        evaluation_result.get("post_qualifying_experience_years", "N/A")
                                    )
                                with col3:
                                    st.metric(
                                        "Min. Experience Required", 
                                        evaluation_result.get("minimum_experience_required", "N/A")
                                    )
                            
                            # Additional metrics row
                            if "education_requirement_met" in evaluation_result:
                                col1, col2 = st.columns(2)
                                with col1:
                                    education_status = evaluation_result.get("education_requirement_met", "Unknown")
                                    if education_status.lower() == "yes":
                                        st.success(f" Education Requirement: ✅ {education_status}")
                                    elif education_status.lower() == "no":
                                        st.error(f" Education Requirement: ❌ {education_status}")
                                    else:
                                        st.info(f" Education Requirement: {education_status}")
                                
                                with col2:
                                    if "key_requirements_from_job" in evaluation_result:
                                        st.info(f" Key Requirements: {evaluation_result.get('key_requirements_from_job', 'Not specified')}")
                            
                            # Expandable sections for document review
                            with st.expander(" View Extracted Resume"):
                                st.text_area(" Resume Text", resume_text, height=200, key="resume_display")
                            
                            if cert_text:
                                with st.expander(" View Extracted Certificate"):
                                    st.text_area("Certificate Text", cert_text, height=200, key="cert_display")
                            
                            with st.expander(" View Selected Job Description"):
                                st.text_area(" Job Description", selected_job['description'], height=200, key="job_display")
                          
                        except Exception as e:
                            st.error(f"❌ Error during evaluation: {str(e)}")
                            logger.error(f"Evaluation error: {e}")
            
            else:
                st.info("Please upload the candidate's documents to proceed with evaluation")
    
    # Reset button
    if st.session_state.jobs_extracted:
        st.markdown("---")
        if st.button("🔄 Start Over (Upload New Job Advertisement)", type="secondary", use_container_width=True):
            st.session_state.jobs_extracted = False
            st.session_state.available_jobs = []
            st.session_state.selected_job = None
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(":blue[*ZRA-AI Innovations © 2025*] ")

if __name__ == "__main__":
    main()
