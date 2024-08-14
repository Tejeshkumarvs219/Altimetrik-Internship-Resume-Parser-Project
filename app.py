import os
import re
import fitz
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, redirect, url_for, flash

nlp = spacy.load('en_core_web_sm')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    pdf_document = fitz.open(file_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    print(f"Extracted text from {file_path}:\n{text[:1000]}...\n") 
    return text

def extract_fields(text):
    doc = nlp(text)
    skills = []
    experience = []
    education = []
    years_of_experience = None
    qualifications = []
    print("Entities found:")
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")
        if ent.label_ in {"PRODUCT", "ORG"}: 
            skills.append(ent.text)
        elif ent.label_ in {"DATE"}:
            experience.append(ent.text)
            # Extract years of experience from text
            match = re.search(r'(\d+)\s*\+?\s*years', ent.text.lower())
            if match:
                years_of_experience = int(match.group(1))

        elif ent.label_ in {"GPE", "FAC"}:
            education.append(ent.text)
        elif ent.label_ == "DEGREE":
            qualifications.append(ent.text)
    print(f"Extracted fields:\nSkills: {skills}\nExperience: {experience}\nEducation: {education}\nYears of Experience: {years_of_experience}\n")
    return {
        "skills": skills,
        "experience": experience,
        "education": education,
        "years_of_experience": years_of_experience,
        "qualifications": qualifications
    }

def match_fields(resume_fields, job_fields):
    if not resume_fields['skills'] or not job_fields['skills']:
        print("No skills to match.")
        return 0.0

    vectorizer = TfidfVectorizer()
    try:
        resume_vector = vectorizer.fit_transform([" ".join(resume_fields['skills'])])
        job_vector = vectorizer.transform([" ".join(job_fields['skills'])])
        similarity = cosine_similarity(resume_vector, job_vector)[0][0]
    except ValueError as e:
        print(f"Error in vectorization or similarity calculation: {e}")
        similarity = 0.0

    # Check experience match
    job_exp_required = job_fields['years_of_experience']
    resume_exp = resume_fields['years_of_experience']
    experience_match = False

    if job_exp_required is not None:
        if resume_exp is not None and resume_exp >= job_exp_required:
            experience_match = True
    else:
        experience_match = True  # If job does not specify experience, assume it matches

    # Check education match
    job_qualifications = set(job_fields['qualifications'])
    resume_qualifications = set(resume_fields['qualifications'])
    education_match = job_qualifications.issubset(resume_qualifications)

    print(f"Skills similarity score: {similarity}, Experience match: {experience_match}, Education match: {education_match}")
    return similarity*100 if experience_match and education_match else 0.0

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'resumes' not in request.files or 'job' not in request.files:
        flash('No file part')
        return redirect(request.url)
    resume_files = request.files.getlist('resumes')
    job_file = request.files['job']
    if not resume_files or job_file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if job_file and allowed_file(job_file.filename):
        job_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job.pdf')
        job_file.save(job_path)
        job_text = extract_text_from_pdf(job_path)
        
        if not job_text.strip():
            flash('Job description document is empty or contains only stop words.')
            return redirect(request.url)
        
        job_fields = extract_fields(job_text)
        
        match_results = []
        for resume_file in resume_files:
            if resume_file and allowed_file(resume_file.filename):
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
                resume_file.save(resume_path)
                resume_text = extract_text_from_pdf(resume_path)
                
                if not resume_text.strip():
                    flash(f'Resume {resume_file.filename} is empty or contains only stop words.')
                    continue
                
                resume_fields = extract_fields(resume_text)
                match_score = match_fields(resume_fields, job_fields)
                match_results.append((resume_file.filename, match_score))
        
        # Sort by match score in descending order
        match_results.sort(key=lambda x: x[1], reverse=True)       
        return render_template('result.html', results=match_results)
    else:
        flash('Invalid job description file type')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
