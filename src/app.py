import streamlit as st
import requests
import os
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from google import genai

load_dotenv()
COLAB_API_URL = os.getenv("COLAB_API_URL", "").rstrip("/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="Document Intelligence", layout="wide")

# Sidebar for document ingestion
st.sidebar.title("Document Management")
st.sidebar.markdown("Upload new documents to the retrieval system.")
uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

if st.sidebar.button("Process Document"):
    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{COLAB_API_URL}/ingest", files=files)
                
                if response.status_code != 200:
                    st.sidebar.error(f"Backend Error: {response.text}")
                else:
                    st.sidebar.success("Document processed successfully.")
            except Exception as e:
                st.sidebar.error(f"System error: {str(e)}")
    else:
        st.sidebar.warning("Please select a file to upload.")

# Main area for search and QA
st.title("Document Retrieval and QA")
st.markdown("Submit a query to retrieve relevant pages and generate an answer.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display images if they exist in the assistant's message
        if "images" in message and message["images"]:
            cols = st.columns(len(message["images"]))
            for idx, img_data in enumerate(message["images"]):
                with cols[idx]:
                    st.image(
                        img_data["img"], 
                        caption=img_data["caption"], 
                        use_container_width=True
                    )

# Accept user input
if query := st.chat_input("Enter your query:"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant pages..."):
            try:
                response = requests.post(
                    f"{COLAB_API_URL}/search",
                    json={"query": query, "top_k": 2}
                )
                
                if response.status_code != 200:
                    st.error(f"Backend Error: {response.text}")
                else:
                    search_results = response.json().get("results", [])

                    if not search_results:
                        st.warning("No relevant pages found.")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "No relevant pages found."
                        })
                    else:
                        st.success("Pages retrieved.")
                        
                        pil_images = []
                        saved_images_for_state = []
                        
                        for res in search_results:
                            img_bytes = base64.b64decode(res["image_base64"])
                            img = Image.open(BytesIO(img_bytes))
                            pil_images.append(img)
                            saved_images_for_state.append({
                                "img": img,
                                "caption": f"Page {res['page_number']} (Score: {res['score']:.2f})"
                            })

                        with st.spinner("Analyzing documents..."):
                            prompt = f"Answer the user's question using ONLY the provided document pages. Extract information carefully from tables or charts. Question: {query}"
                            contents = [prompt] + pil_images
                            
                            gemini_response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=contents
                            )
                            
                            answer = gemini_response.text
                            st.markdown(answer)
                            
                            # Display images below the answer
                            st.subheader("Source Documents")
                            cols = st.columns(len(saved_images_for_state))
                            for idx, img_data in enumerate(saved_images_for_state):
                                with cols[idx]:
                                    st.image(
                                        img_data["img"], 
                                        caption=img_data["caption"], 
                                        use_container_width=True
                                    )
                            
                            # Save assistant response to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer,
                                "images": saved_images_for_state
                            })

            except Exception as e:
                st.error(f"System error: {str(e)}")