import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Multi-Modal QA Bot", layout="wide")
st.title("📄 Multi-Modal Document RAG")

query = st.text_input("Ask a question about your documents:")

if st.button("Search & Answer"):
    if not query:
        st.warning("Please enter a query.")
    else:
        st.info("This is a placeholder. Soon this will call our Colab API and Gemini!")
        # We will add the actual logic in the next step.