
import streamlit as st
import joblib
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="News Intelligence AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD TRAINED MODEL ----------------
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "fake_news_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ---------------- CUSTOM CSS DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #101018 0%, #171426 55%, #101820 100%);
    color: #F1F5F9;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(16, 16, 24, 0.8);
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #211B38, #172B3B);
    border: 1px solid #38314F;
    border-radius: 22px;
    padding: 34px;
    margin-bottom: 24px;
}

.eyebrow {
    color: #67E8F9;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
}

.hero h1 {
    color: #FFFFFF;
    font-size: clamp(30px, 5vw, 48px);
    font-weight: 800;
    margin: 12px 0;
}

.gradient-text {
    background: linear-gradient(90deg, #C4B5FD, #67E8F9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #B9B7CC;
    font-size: 16px;
    line-height: 1.7;
}

.section-title {
    color: #F1F5F9;
    font-size: 21px;
    font-weight: 700;
    margin-top: 14px;
}

.panel {
    background: rgba(34, 30, 51, 0.88);
    border: 1px solid #3B3452;
    border-radius: 18px;
    padding: 24px;
}

div[data-testid="stTextArea"] textarea {
    background: #151522;
    color: #F8FAFC;
    border: 1px solid #49415F;
    border-radius: 12px;
    font-size: 15px;
    line-height: 1.7;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #A78BFA;
    box-shadow: 0 0 0 1px #A78BFA;
}

.stButton > button {
    background: linear-gradient(90deg, #8B5CF6, #6366F1);
    color: white;
    border: 0;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 700;
    width: 100%;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border: 1px solid #67E8F9;
    color: white;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.25);
}

.result-card {
    background: linear-gradient(135deg, #211B38, #172B3B);
    border: 1px solid #49415F;
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
}

.result-label {
    color: #A5B4FC;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

.result-value {
    font-size: 30px;
    font-weight: 800;
    margin: 10px 0;
}

.small-note {
    color: #A5A3B8;
    font-size: 13px;
    line-height: 1.7;
}

div[data-testid="stMetric"] {
    background: #211B38;
    border: 1px solid #3B3452;
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #B9B7CC;
}

hr {
    border-color: #38314F;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ INTELLIGENT TEXT ANALYSIS</div>
    <h1>NEWS <span class="gradient-text">INTELLIGENCE</span></h1>
    <p>
        An AI-powered text classification system that learns patterns
        from labelled news articles to predict whether an article is
        likely fake or likely real.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
st.markdown('<div class="section-title">◈ Analyse a news article</div>',
            unsafe_allow_html=True)

st.write(
    "Paste a headline or article below. The model will analyse its text patterns."
)

news_text = st.text_area(
    "News article",
    height=230,
    placeholder="Paste your news headline or article here...",
    label_visibility="collapsed"
)

col1, col2 = st.columns([3, 1])

with col1:
    analyse = st.button("✦  RUN AI ANALYSIS", use_container_width=True)

with col2:
    clear = st.button("↻  CLEAR TEXT", use_container_width=True)

if clear:
    st.session_state["news_input"] = ""
    st.rerun()

# ---------------- PREDICTION ----------------
if analyse:
    if not news_text.strip():
        st.warning("Please enter a headline or article before analysing.")
    else:
        with st.spinner("Analysing text patterns..."):
            features = vectorizer.transform([news_text])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

        st.markdown("---")
        st.markdown(
            '<div class="section-title">◈ Analysis result</div>',
            unsafe_allow_html=True
        )

        result_col, score_col = st.columns([1.5, 1])

        with result_col:
            if prediction == 0:
                result = "LIKELY FAKE"
                accent = "#FDA4AF"
                icon = "⚠"
            else:
                result = "LIKELY REAL"
                accent = "#67E8F9"
                icon = "✓"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">MODEL CLASSIFICATION</div>
                <div class="result-value" style="color:{accent}">
                    {icon} {result}
                </div>
                <div class="small-note">
                    Classification based on patterns learned from the training dataset.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with score_col:
            score = float(probabilities[prediction]) * 100

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.metric("Model score for predicted class", f"{score:.2f}%")
            st.progress(score / 100)
            st.markdown(
                '<div class="small-note">This is a model score, not verified factual confidence.</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.info(
            "Important: This model does not independently fact-check news. "
            "A prediction is not proof that an article is true or false."
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:12px;">
    <div style="color:#A78BFA; font-weight:700;">
        NEWS INTELLIGENCE AI
    </div>
    <div class="small-note">
        Built with Python · TF-IDF · Logistic Regression · Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
