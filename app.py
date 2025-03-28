import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit UI
st.title("Spam vs Ham Classifier 🚀")
st.write("Enter a message below to classify it as Spam or Ham.")

# User input
user_input = st.text_area("Enter message:", "")

if st.button("Classify"):
    if user_input:
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
        st.warning("⚠️ Please enter some text to classify.")