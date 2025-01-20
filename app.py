import pickle as pkl
import streamlit as st
from nltk.stem import PorterStemmer
import string
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from nltk.corpus import stopwords
import warnings
import os
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
stemmer = PorterStemmer() # this is used to shorten a word back to its root word

llm_model = ChatGroq(
    api_key=api_key,
    model="llama-3.2-3b-preview",
)

def second_workflow(text): # this function uses the random_forest classes to classify the mail
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
    return response

