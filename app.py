import streamlit as st
import pickle
import urllib.request
import os

# Define the GitHub URLs of the files
MODEL_URL = "https://raw.githubusercontent.com/HolaDav/Machine-Learning-Projects/main/model.pkl"
VECTORIZER_URL = "https://raw.githubusercontent.com/HolaDav/Machine-Learning-Projects/main/vectorizer.pkl"

# Download files if they don't exist
if not os.path.exists("model.pkl"):
    urllib.request.urlretrieve(MODEL_URL, "model.pkl")

if not os.path.exists("vectorizer.pkl"):
    urllib.request.urlretrieve(VECTORIZER_URL, "vectorizer.pkl")

# Load the model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit UI
st.title("Spam vs Ham Classifier 🚀")
st.write("Enter a message below to classify it as Spam or Ham.")

user_input = st.text_area("Enter message:", "")

if st.button("Classify"):
    if user_input:
        input_transformed = vectorizer.transform([user_input])
        prediction = model.predict(input_transformed)[0]

        if prediction == "ham":
            st.success("✅ This is a Ham message!")
        else:
            st.error("🚨 This is a Spam message!")
    else:
        st.warning("⚠️ Please enter some text to classify.")
