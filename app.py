import streamlit as st
import pickle

# Load saved models
with open("spam_models.pkl", "rb") as file:
    data = pickle.load(file)

models = data["models"]
accuracy_scores = data["accuracy"]

# Title
st.title("📩 AI-Powered Spam Detection System")

# Model Selection
selected_model_name = st.selectbox("🔍 Choose a Model", list(models.keys()))
selected_model = models[selected_model_name]

# Display Accuracy
st.markdown(f"📊 **Model Accuracy:** `{accuracy_scores[selected_model_name]:.2%}`")

# Input Text
st.subheader("💬 Check if a Message is Spam")
message = st.text_area("Enter a message:")

# Predict Button
if st.button("Check Spam"):
    if message:
        prediction = selected_model.predict([message])[0]
        result = "🚫 **Spam**" if prediction == 1 else "✅ **Not Spam**"
        st.success(f"Prediction: {result}")
