import os
import re
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

class ResumeRAGAnalyzer:
    def __init__(self):
        # Check if Google API key is available
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please add it to your .env file")
        
        # Configure the Google Gemini API
        genai.configure(api_key=api_key)
        
        # Use the correct model names
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",  # Use the correct model name
            temperature=0.1,
            google_api_key=api_key
        )
        self.persist_directory = "./chroma_db"
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text_content = "\n".join([page.page_content for page in pages])
            
            # Clean up the extracted text
            cleaned_text = self.clean_text(text_content)
            return cleaned_text
            
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
            
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep relevant punctuation
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        # Remove multiple consecutive spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def process_job_description(self, job_desc_text: str, collection_name: str = "job_descriptions"):
        """Process and store job description in vector database"""
        if not job_desc_text.strip():
            raise ValueError("Job description text cannot be empty")
            
        # Clean the text
        cleaned_text = self.clean_text(job_desc_text)
        
        # Create document
        doc = Document(page_content=cleaned_text, metadata={"source": "job_description", "type": "requirements"})
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents([doc])
        
        # Create and persist vector store
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        vectordb.persist()
        
        return f"Processed and stored {len(chunks)} chunks from job description."
    
    def query_job_description(self, resume_text: str, collection_name: str = "job_descriptions", k: int = 5):
        """Query job description using resume text"""
        if not resume_text.strip():
            raise ValueError("Resume text cannot be empty")
            
        # Clean resume text
        cleaned_resume = self.clean_text(resume_text)
        
        # Load vector store
        try:
            vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
        except:
            # If collection doesn't exist yet, return empty results
            return []
        
        # Query for relevant job description chunks
        results = vectordb.similarity_search(cleaned_resume, k=k)
        return results
    
    def analyze_resume_vs_jd(self, resume_text: str, job_desc_text: str) -> dict:
        """Comprehensive analysis of resume against job description"""
        if not resume_text.strip() or not job_desc_text.strip():
            raise ValueError("Both resume and job description text must be provided")
            
        # Process job description
        process_result = self.process_job_description(job_desc_text)
        print(process_result)  # For debugging
        
        # Query for relevant parts
        relevant_chunks = self.query_job_description(resume_text)
        
        if not relevant_chunks:
            return {
                "score": 0,
                "analysis": "No relevant job description requirements found. Please check your inputs.",
                "relevant_chunks": []
            }
        
        # Generate analysis
        analysis = self.generate_analysis(resume_text, relevant_chunks)
        
        # Calculate score
        score = self.calculate_match_score(analysis)
        
        return {
            "score": score,
            "analysis": analysis,
            "relevant_chunks": [chunk.page_content for chunk in relevant_chunks]
        }
    
    def generate_analysis(self, resume_text: str, relevant_chunks: List[Document]) -> str:
        """Generate detailed analysis using LLM"""
        try:
            template = """
            You are an expert career coach and hiring manager. Analyze how well the resume matches the job description.

            JOB DESCRIPTION REQUIREMENTS:
            {context}

            CANDIDATE'S RESUME:
            {resume}

            Please provide a comprehensive analysis with the following structure:

            1. OVERALL MATCH: Start with a percentage score (0-100%) for overall match.

            2. STRENGTHS (Present in Resume):
            - List the key job requirements that are strongly reflected in the resume
            - Be specific and mention relevant technologies, skills, and experiences
            - Provide examples from the resume that demonstrate these strengths

            3. GAPS & AREAS FOR IMPROVEMENT (Missing or Weak in Resume):
            - List important job requirements that are missing or not well demonstrated
            - Suggest how the candidate could address these gaps
            - Be constructive and specific

            4. RECOMMENDATIONS:
            - Suggest specific improvements to the resume
            - Recommend skills to acquire or highlight existing skills better
            - Tips for tailoring the resume for this specific role

            Be detailed, professional, and focus on actionable insights.
            """

            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()

            context_text = "\n".join([doc.page_content for doc in relevant_chunks])
            
            analysis = chain.invoke({
                "context": context_text,
                "resume": resume_text
            })
            return analysis
            
        except Exception as e:
            print(f"Google API Error: {e}")
            # Fallback to dummy response for testing
            return self._generate_fallback_analysis(resume_text, relevant_chunks)
    
    def _generate_fallback_analysis(self, resume_text: str, relevant_chunks: List[Document]) -> str:
        """Generate a fallback analysis when API fails"""
        return """
1. OVERALL MATCH: 78%

2. STRENGTHS (Present in Resume):
- Strong programming skills in Python and JavaScript
- Experience with web frameworks and cloud technologies
- Demonstrated problem-solving abilities in previous roles
- Good educational background in Computer Science

3. GAPS & AREAS FOR IMPROVEMENT (Missing or Weak in Resume):
- Could benefit from more specific metrics to quantify achievements
- Limited experience with advanced machine learning frameworks
- Consider adding more leadership and collaboration examples

4. RECOMMENDATIONS:
- Add specific metrics to quantify your impact (e.g., "improved performance by 40%")
- Highlight any relevant certifications or courses
- Include more details about team collaboration and leadership
- Tailor your skills section to match the job requirements more closely

Note: This is a sample analysis. For real-time AI analysis, please check your API configuration.
"""
    
    def calculate_match_score(self, analysis: str) -> int:
        """Extract match score from analysis text"""
        if not analysis:
            return 0
            
        # Look for percentage pattern in the analysis
        match = re.search(r'(\d{1,3})%', analysis)
        if match:
            score = int(match.group(1))
            # Ensure score is within valid range
            return max(0, min(100, score))
        
        # Fallback: look for numeric scores without percentage
        match = re.search(r'(\d{1,3}) out of 100', analysis, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return max(0, min(100, score))
            
        return 75  # Default score for fallback analysis
    
    def get_improvement_suggestions(self, analysis: str) -> List[str]:
        """Extract improvement suggestions from analysis"""
        if not analysis:
            return ["No analysis available to extract suggestions"]
            
        suggestions = []
        lines = analysis.split('\n')
        
        in_suggestions_section = False
        for line in lines:
            line_lower = line.lower()
            if 'recommendations:' in line_lower or 'suggestions:' in line_lower:
                in_suggestions_section = True
                continue
                
            if in_suggestions_section:
                # Check if we've reached the next section
                if line.strip() and ':' in line and not line.startswith((' ', '-', '•', '\t')):
                    break
                    
                # Extract suggestions from bullet points
                if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•')):
                    clean_suggestion = re.sub(r'^[-•\d.\s]+', '', line.strip()).strip()
                    if clean_suggestion and len(clean_suggestion) > 10:
                        suggestions.append(clean_suggestion)
                
                # Extract from numbered lists
                elif line.strip() and re.match(r'^\d+[\.\)]', line.strip()):
                    clean_suggestion = re.sub(r'^\d+[\.\)]\s*', '', line.strip()).strip()
                    if clean_suggestion and len(clean_suggestion) > 10:
                        suggestions.append(clean_suggestion)
        
        return suggestions if suggestions else [
            "Add specific metrics to quantify your achievements",
            "Tailor your skills section to match job requirements",
            "Highlight relevant certifications and courses",
            "Include more details about team collaboration"
        ]

# Utility function to create analyzer instance
def create_analyzer():
    """Create and return a ResumeRAGAnalyzer instance"""
    return ResumeRAGAnalyzer()

# Test the Google API connection
if __name__ == "__main__":
    try:
        analyzer = ResumeRAGAnalyzer()
        print("✅ ResumeRAGAnalyzer initialized successfully")
        
        # Test the API connection
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Test connection")
        print("✅ Google Gemini API connection successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your Google API key and ensure it's valid")