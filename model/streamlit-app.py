# import streamlit as st
# import joblib
# from pathlib import Path
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB

# count_vector = CountVectorizer()
# naive_bayes = MultinomialNB()

# model_directory = Path().absolute().parent / "artifact"
# model_file_path = model_directory / "modelsave.pkl"

# model_from_joblib = joblib.load(model_file_path)

# st.title("Predicting if the message is spam or ham")

# message = st.text_input("Enter the message")

# submit = st.button('prediction')

# if submit:
#     # Vectorize the input message
#     vectorized_message = count_vector.transform([message])
#     # Predict using the loaded model
#     prediction = naive_bayes.predict(vectorized_message)[0]
#     # Translate the numerical prediction to a label
#     result = 'spam' if prediction == 1 else 'ham'
#     # Write the result
#     st.write(f'The given message is classified as: {result}')

import streamlit as st
import joblib
from pathlib import Path

# Function to load the model and vectorizer from the file
def load_model_vectorizer(file_path):
    return joblib.load(file_path)

# Function to classify an email message
def classify_email(model, vectorizer, message):
    vectorized_message = vectorizer.transform([message])
    prediction = model.predict(vectorized_message)[0]
    return 'spam' if prediction == 1 else 'ham'

# Path to the model file
model_directory = Path().absolute().parent / "artifact"
model_file_path = model_directory / "modelsave.pkl"

# Load the model and vectorizer
spam_model, email_vectorizer = load_model_vectorizer(model_file_path)

# Streamlit UI
st.title("Spam or Ham Message Classifier")

# Input field for the user message
user_message = st.text_area("Enter the message you want to classify:")

# Button for prediction
if st.button("Predict"):
    # Make a prediction
    classification = classify_email(spam_model, email_vectorizer, user_message)
    # Display the result
    st.write(f'The given message is classified as: **{classification}**')