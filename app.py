import streamlit as st
from predict import predict_sentiment

st.title("📝 Product Review Sentiment Analyzer")

user_input = st.text_area("Enter your product review:")

if st.button("Analyze Sentiment"):
    if user_input:
        prediction = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {prediction}")
    else:
        st.write("Please enter a review first.")
