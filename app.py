import streamlit as st
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    body {
        background-color: #f4f6f9;
    }
    .main-title {
        font-size: 38px;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
        background-color: #1f77b4;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📰 Fake News Detection System</div>', unsafe_allow_html=True)

st.sidebar.header("Project Info")
st.sidebar.write("**Model:** LinearSVC")
st.sidebar.write("**Vectorizer:** TF-IDF")
st.sidebar.write("**Developed by:** Leisha Jain")




if st.button("🔍 Analyze News"):

    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        vect_text = vectorizer.transform([input_text])
        prediction = model.predict(vect_text)
        decision = model.decision_function(vect_text)
        confidence = abs(decision[0])

        st.markdown("---")

        if prediction[0] == 0:
            st.success("✅ This News Appears to be REAL")
        else:
            st.error("🚨 This News Appears to be FAKE")

        st.info(f"🔎 Confidence Score: {round(confidence, 2)}")

        st.markdown("---")

        st.caption("⚠️ This prediction is based on machine learning analysis and may not be 100% accurate.")


