import pickle as pkl
import streamlit as st
from nltk.stem import PorterStemmer
import string
from langchain_groq import ChatGroq
from nltk.corpus import stopwords
import warnings
import os
nltk.download('stopwords')
# Suppress warnings
warnings.filterwarnings("ignore")

api_key = st.secrets["API_KEY"]

def pickle_loader(path_of_pickle_file: str):
    with open(path_of_pickle_file, "rb") as model_obj:
        output = pkl.load(model_obj)
    return output

vectorizer = pickle_loader(r"Vectorizer.pkl")
model = pickle_loader(r"RandomForest_Model.pkl")
label_encoder = pickle_loader(r"LabelEncoder.pkl")

stopwords_set = set(stopwords.words("english"))
stemmer = PorterStemmer()


llm_model = ChatGroq(
    api_key=api_key,
    model="llama-3.2-3b-preview",
)

# Function to classify email content
def second_workflow(text):
    email_text = (
        text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    )
    email_text = [
        stemmer.stem(word) for word in email_text if word not in stopwords_set
    ]
    email_text = " ".join(email_text)
    email_corpus = [email_text]
    x_email = vectorizer.transform(email_corpus)
    predicted = model.predict(x_email)
    actual_label = label_encoder.inverse_transform(predicted)
    response = f"The Spam Detector has identified the email as a {actual_label[0]} message."
    return response, actual_label

# Streamlit UI
st.title("Email Spam Detection and Analysis")

# Input fields for email subject and body
subject = st.text_input("Enter the email subject:")
body = st.text_area("Enter the email body:")

# Concatenate subject and body
entire_mail = f"{subject} {body}"

# Button to process the email
if st.button("Analyze Email"):
    if subject.strip() and body.strip():
        st.write("Processing your request...")
        try:
            # Use the second_workflow function
            workflow_output, actual_label = second_workflow(entire_mail)
            st.write(workflow_output)

            # Generate LLM response based on the classification
            llm_response = llm_model.invoke(f"""
            The provided email is a {actual_label[0]} mail.
            If the type is either ransomware, phishing, or BEC, extract spam-like keywords from the following email: {entire_mail}.
            If it is a 'not-harmful' mail, state clearly that the mail is clean and provide a reasonable explanation for why it is classified as such.
            
            Always start the output with "The provided email is [mail type] mail."
            Ensure the output is concise and easy to interpret at first glance.
            
            Note: The only types of mail are: BEC, phishing, ransomware, and not-harmful mail.
            """)

            # Display LLM response
            st.write("### Analysis Result:")
            st.write(llm_response)
        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
    else:
        st.warning("Please fill in both the subject and body fields!")
