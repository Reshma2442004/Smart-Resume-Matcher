import fitz
import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy
nlp = spacy.load("en_core_web_sm")
import spacy
import spacy.cli

# Ensure the model is downloaded
spacy.cli.download("en_core_web_sm")

# Then load the model
nlp = spacy.load("en_core_web_sm")

# Page settings
st.set_page_config(page_title="Smart Resume Matcher", page_icon="📄", layout="wide")

#  Custom CSS Styling
st.markdown("""
    <style>
        body {
            background-color: #eef2f7;
        }
        .main {
            text-align: center;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #0077cc;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
        }
        .stButton>button:hover {
            background-color: #005c99;
        }
        .title-style {
            background-color: #003366;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .card {
            padding: 20px;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.08);
            margin: 20px auto;
            width: 90%;
        }
        .section-title {
            color: #0d47a1;
            margin-top: 1.5rem;
        }
        .home-header {
            background: linear-gradient(90deg, #003366, #004080);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
        }
        .features-container {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 2rem;
            flex-wrap: wrap;
        }
        .feature-box {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            width: 300px;
        }
   .stDownloadButton > button {
            background-color: #0077cc;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
            border: none;
            font-weight: bold;
        }
        .stDownloadButton > button:hover {
            background-color: #005c99;
        }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def extract_text_from_pdf(uploaded_file):
    pdf_text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()
    return pdf_text

def get_match_score(resume_text, jd_text):
    processed_resume = preprocess(resume_text)
    processed_jd = preprocess(jd_text)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([processed_resume, processed_jd])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

def extract_skills(text):
    keywords = [
        "python", "sql", "machine learning", "deep learning", "power bi", "tableau",
        "data science", "excel", "nlp", "scikit-learn", "tensorflow", "xgboost"
    ]
    text = text.lower()
    return [skill for skill in keywords if skill in text]

# --- Sidebar Navigation ---
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://cdn-icons-png.flaticon.com/512/3135/3135755.png" width="100" style="margin-bottom: 10px;" />
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Smart Resume Matcher")
nav = st.sidebar.radio("Navigation", ["Home", "Resume Matcher", "About"])

# --- Pages ---

if nav == "Home":
    st.markdown("""
        <div class='home-header'>
            <h1>Welcome to Smart Resume Matcher</h1>
            <p>AI-powered tool to match resumes with job descriptions using NLP</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='features-container'>
            <div class='feature-box'>
                <img src="https://cdn-icons-png.flaticon.com/512/4781/4781517.png" width="80"/>
                <h4>PDF Resume Parsing</h4>
                <p>Extract text directly from PDF resumes with high accuracy.</p>
            </div>
            <div class='feature-box'>
                <img src="https://cdn-icons-png.flaticon.com/512/3063/3063826.png" width="80"/>
                <h4>AI Skill Matching</h4>
                <p>Get skill overlap and similarity score using NLP and cosine similarity.</p>
            </div>
            <div class='feature-box'>
                <img src="https://cdn-icons-png.flaticon.com/512/854/854878.png" width="80"/>
                <h4>Visual Dashboard</h4>
                <p>Clean, modern UI with skill comparison and resume insights.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br><center><h4 style='color: #0d47a1;'>Start matching now from the 'Resume Matcher' tab in the sidebar</h4></center>", unsafe_allow_html=True)

elif nav == "Resume Matcher":
    st.markdown("<div class='title-style'><h1>Resume Matching Dashboard</h1></div>", unsafe_allow_html=True)

    st.markdown("### Step 1: Upload Resume & Paste Job Description", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        uploaded_resume = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

    with col2:
        jd_input = st.text_area("Paste Job Description Here", height=250)

    st.markdown("### Step 2: Run Matching", unsafe_allow_html=True)

    if st.button("Match Now"):
        if uploaded_resume and jd_input:
            resume_text = extract_text_from_pdf(uploaded_resume)
            match_score = get_match_score(resume_text, jd_input)

            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(jd_input)

            matched_skills = list(set(resume_skills) & set(jd_skills))
            missing_skills = list(set(jd_skills) - set(resume_skills))

            st.markdown(f"<div class='card'><h2>Match Score: {match_score}%</h2></div>", unsafe_allow_html=True)

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("### Matched Skills", unsafe_allow_html=True)
                if matched_skills:
                    st.success(", ".join(matched_skills))
                else:
                    st.warning("No matched skills found.")

            with col4:
                st.markdown("### Missing Skills", unsafe_allow_html=True)
                if missing_skills:
                    st.error(", ".join(missing_skills))
                else:
                    st.info("All required skills matched!")

            with st.expander("View Resume Text (Extracted)"):
                st.write(resume_text)
                # Generate Downloadable Report
            report = f"""
            Smart Resume Matcher Report
            ---------------------------
            Match Score: {match_score}%

            Matched Skills:
            {', '.join(matched_skills) if matched_skills else 'None'}

            Missing Skills:
            {', '.join(missing_skills) if missing_skills else 'None'}
            """

            st.download_button(
                label=" Download Report",
                data=report,
                file_name="resume_match_report.txt",
                mime="text/plain"
            )

        else:
            st.error("Please upload both resume and job description.")

elif nav == "About":
    st.markdown("<div class='title-style'><h1>About</h1></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='card'>
            <h4>Smart Resume Matcher</h4>
            <p>
                This project uses Natural Language Processing (NLP) and machine learning to analyze and score resumes 
                against job descriptions.
            </p>
            <ul>
                <li>Skill Matching with TF-IDF + Cosine Similarity</li>
                <li>PDF Resume Extraction</li>
                <li>Real-time Streamlit UI</li>
            </ul>
            <p>Created by <strong>@Reshma_Gade Email - reshmagade.intern@gmail.com</strong></p>
        </div>
    """, unsafe_allow_html=True)
