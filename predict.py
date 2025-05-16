import joblib

# Load model and vectorizer
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    return model.predict(text_vector)[0]
