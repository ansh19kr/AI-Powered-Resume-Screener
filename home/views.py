from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample job description
JOB_DESCRIPTION = """
We are looking for a Python Fresher who is passionate about data analysis and numerical computing. 
The ideal candidate should have:
- Strong understanding of Python programming.
- Hands-on experience with NumPy and Pandas for data manipulation and analysis.
- Basic knowledge of data structures and algorithms.
- Ability to work with CSV, Excel, and JSON files using Pandas.
- Good problem-solving skills and logical thinking.
- Enthusiasm for learning and growing in a data-focused role.

Freshers with relevant projects or certifications in NumPy and Pandas are highly encouraged to apply!
"""


def extract_text_from_resume(resume_file):
    """Extract text from a resume file based on its format."""
    
    file_name = resume_file.name.lower()  # Ensure we're getting the file name
    
    # Handle .docx files
    if file_name.endswith('.docx'):
        text = docx2txt.process(resume_file)

    # Handle .pdf files
    elif file_name.endswith('.pdf'):
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(resume_file)  # Read directly from the file
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")

    # Handle .txt files
    elif file_name.endswith('.txt'):
        try:
            text = resume_file.read().decode('utf-8', errors='ignore')  # Read directly from the file object
        except Exception as e:
            print(f"Error reading TXT file: {e}")
    
    else:
        return "Unsupported file format"

    return text if text.strip() else "No text extracted"

def calculate_similarity(resume_text, job_description):
    """Calculate similarity between resume and job description."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity_score[0][0] * 100  # Convert to percentage

def resume_screening(request):
    """Handle resume upload and screening."""
    if request.method == 'POST' and request.FILES.get('resume'):
        uploaded_file = request.FILES['resume']  # Get the uploaded file
        
        # Extract text and calculate match percentage
        resume_text = extract_text_from_resume(uploaded_file)  # Pass the file object
        match_percentage = calculate_similarity(resume_text, JOB_DESCRIPTION)
        
        return render(request, 'result.html', {'match': match_percentage})

    return render(request, 'upload.html')

