import streamlit as st
import pickle
import os

# Check if the model and vectorizer files exist before loading
if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
else:
    st.error("Model files not found! Make sure `model.pkl` and `vectorizer.pkl` are in the same directory.")

# Streamlit UI
st.title("Spam vs Ham Classifier 🚀")
st.write("Enter a message below to classify it as Spam or Ham.")

# User input
user_input = st.text_area("Enter message:", "")

if st.button("Classify"):
    if user_input:
        if "model" in globals() and "vectorizer" in globals():
            # Transform input text
            input_transformed = vectorizer.transform([user_input])

            # Predict
            prediction = model.predict(input_transformed)[0]

            # Display result
            if prediction == "ham":
                st.success("✅ This is a Ham message!")
            else:
                st.error("🚨 This is a Spam message!")
        else:
            st.error("Model is not loaded. Please check your files.")
    else:
        st.warning("⚠️ Please enter some text to classify.")
