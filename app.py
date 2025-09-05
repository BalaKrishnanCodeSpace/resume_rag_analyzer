import streamlit as st
import tempfile
import os
import time
from datetime import datetime
from rag_core import ResumeRAGAnalyzer

# Page configuration
st.set_page_config(
    page_title="Resume RAG Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return ResumeRAGAnalyzer()

def safe_file_cleanup(file_path):
    """Safely delete a file with retries and error handling"""
    if file_path and os.path.exists(file_path):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                os.unlink(file_path)
                break
            except (PermissionError, OSError):
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                else:
                    print(f"Warning: Could not delete temporary file: {file_path}")

def extract_section(text: str, section_name: str) -> str:
    """Extract a specific section from the analysis text"""
    if not text:
        return f"No content available for {section_name}"
        
    lines = text.split('\n')
    section_lines = []
    in_section = False
    
    for line in lines:
        if section_name.lower() in line.lower() and ':' in line:
            in_section = True
            continue
        if in_section:
            if line.strip() and ':' in line and not line.startswith((' ', '-', '•', '\t')):
                break
            if line.strip():
                section_lines.append(line)
    
    result = '\n'.join(section_lines).strip()
    return result if result else f"No specific content found for {section_name}"

def main():
    st.title("📄 AI Resume & Job Description Analyzer")
    st.markdown("""
    Upload your resume and a job description to get an AI-powered analysis of how well you match the role.
    Get detailed feedback on strengths, gaps, and improvement suggestions.
    """)
    
    # Initialize analyzer
    try:
        analyzer = get_analyzer()
    except ValueError as e:
        st.error(f"❌ Configuration Error: {e}")
        st.info("Please make sure you have a GOOGLE_API_KEY in your .env file")
        return
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
        return
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Job Description")
        job_desc_file = st.file_uploader(
            "Upload Job Description PDF",
            type="pdf",
            key="jd_upload",
            help="Upload the job description you want to match against"
        )
        
    with col2:
        st.subheader("👤 Your Resume")
        resume_file = st.file_uploader(
            "Upload Your Resume PDF",
            type="pdf",
            key="resume_upload",
            help="Upload your resume in PDF format"
        )
    
    # Alternative text input
    st.subheader("📝 Or paste text directly")
    text_col1, text_col2 = st.columns(2)
    
    with text_col1:
        job_desc_text = st.text_area(
            "Paste Job Description Text",
            height=150,
            placeholder="Copy and paste the job description text here...",
            help="Alternative to file upload"
        )
    
    with text_col2:
        resume_text = st.text_area(
            "Paste Resume Text",
            height=150,
            placeholder="Copy and paste your resume text here...",
            help="Alternative to file upload"
        )
    
    # Analysis button
    if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
        if not any([job_desc_file, job_desc_text]) or not any([resume_file, resume_text]):
            st.error("Please provide both a job description and a resume!")
            return
        
        with st.spinner("🔍 Analyzing your documents..."):
            job_desc_path = None
            resume_path = None
            
            try:
                # Handle file uploads
                if job_desc_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_jd:
                        tmp_jd.write(job_desc_file.getvalue())
                        job_desc_path = tmp_jd.name
                    job_desc_content = analyzer.extract_text_from_pdf(job_desc_path)
                    st.success("✓ Job description processed")
                else:
                    job_desc_content = job_desc_text
                    st.success("✓ Job description text used")
                
                if resume_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_resume:
                        tmp_resume.write(resume_file.getvalue())
                        resume_path = tmp_resume.name
                    resume_content = analyzer.extract_text_from_pdf(resume_path)
                    st.success("✓ Resume processed")
                else:
                    resume_content = resume_text
                    st.success("✓ Resume text used")
                
                # Perform analysis
                results = analyzer.analyze_resume_vs_jd(resume_content, job_desc_content)
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Score display
                score = results['score']
                st.subheader(f"Overall Match Score: {score}%")
                
                # Progress bar for visual effect
                progress_color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                st.progress(score/100, text=f"Match Strength: {score}%")
                
                # Analysis sections
                st.markdown("---")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Full Analysis", "✅ Strengths", "⚠️ Gaps", "💡 Recommendations"])
                
                with tab1:
                    st.subheader("Detailed Analysis")
                    st.markdown(results['analysis'])
                
                with tab2:
                    strengths_section = extract_section(results['analysis'], "STRENGTHS")
                    if strengths_section and "No specific content" not in strengths_section:
                        st.markdown(strengths_section)
                    else:
                        st.info("No specific strengths identified in the analysis.")
                
                with tab3:
                    gaps_section = extract_section(results['analysis'], "GAPS")
                    if gaps_section and "No specific content" not in gaps_section:
                        st.markdown(gaps_section)
                    else:
                        st.info("No specific gaps identified in the analysis.")
                
                with tab4:
                    suggestions = analyzer.get_improvement_suggestions(results['analysis'])
                    st.subheader("Key Recommendations")
                    if suggestions and len(suggestions) > 0:
                        for i, suggestion in enumerate(suggestions, 1):
                            st.markdown(f"**{i}.** {suggestion}")
                    else:
                        st.info("No specific recommendations found. Check the full analysis for insights.")
                
                # Download analysis
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                analysis_text = f"Resume Analysis Report\nGenerated: {datetime.now()}\n\nMatch Score: {score}%\n\n{results['analysis']}"
                
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=analysis_text,
                    file_name=f"resume_analysis_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("Please check that your PDF files are valid and not password protected.")
            
            finally:
                # Safely clean up temporary files
                safe_file_cleanup(job_desc_path)
                safe_file_cleanup(resume_path)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Resume RAG Analyzer** 🤖
    
    This AI-powered tool analyzes how well your resume matches a job description using advanced RAG (Retrieval-Augmented Generation) technology.
    
    **How it works:**
    1. 📋 Upload job description and resume
    2. 🔍 AI analyzes requirements vs your skills
    3. 📊 Get detailed feedback and match score
    
    **Features:**
    - ✅ Match percentage scoring
    - ✅ Strength identification
    - ✅ Gap analysis
    - ✅ Improvement suggestions
    - 📥 Downloadable reports
    """)
    
    st.header("⚙️ Settings")
    if st.button("Clear Cache & Refresh", help="Clear all cached data and refresh the analyzer"):
        st.cache_resource.clear()
        st.success("Cache cleared! Refreshing...")
        st.rerun()
    
    st.header("🔧 Technical Info")
    st.code("Powered by:\n- Google Gemini AI\n- LangChain RAG\n- Streamlit UI", language="plaintext")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, LangChain, and Google Gemini AI")

if __name__ == "__main__":
    main()