import gradio as gr
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

# Define the prediction function for Gradio
def classify_message(message):
    return classify_email(spam_model, email_vectorizer, message)

# Create Gradio interface
iface = gr.Interface(
    fn=classify_message, 
    inputs="text", 
    outputs="text",
    title="Spam or Ham Message Classifier",
    description="Enter a message to classify if it is spam or ham.",
    examples=[["Free entry in 2 a wkly comp to win FA Cup fina..."],
              ["Hello, how are you doing today?"],
              ["Claim your free prize now by clicking this link!"]]
)

# Launch the app
if __name__ == "__main__":
    iface.launch()