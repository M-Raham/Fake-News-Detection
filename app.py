import streamlit as st
import joblib

# Load trained model & vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit App
st.title("📰 Fake News Detector")
st.write("Enter a news article below and check if it's real or fake.")

# User Input
news_text = st.text_area("Enter News Article:", height=200)

if st.button("Check News"):
    if news_text.strip():
        # Convert input text into numerical form
        input_tfidf = vectorizer.transform([news_text])

        # Predict using the trained model
        prediction = model.predict(input_tfidf)[0]

        # Show Result
        if prediction == 0:
            st.error("🚨 Fake News Detected!")
        else:
            st.success("✅ This News is Real!")
    else:
        st.warning("⚠️ Please enter some text before checking.")
