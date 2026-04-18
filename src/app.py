import streamlit as st
import requests
import os
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
COLAB_API_URL = os.getenv("COLAB_API_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="Multi-Modal QA Bot", layout="wide")
st.title("📄 Multi-Modal Document RAG")
st.markdown("Ask a question, and the system will retrieve the exact visual pages from Qdrant and use Gemini to read the charts/tables to answer it.")

query = st.text_input("Ask a question about your documents:")

if st.button("Search & Answer"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("🔍 Searching document vectors via Colab..."):
            try:
                # 1. Ask Colab to search Qdrant using ColPali
                response = requests.post(
                    f"{COLAB_API_URL}/search",
                    json={"query": query, "top_k": 2} # Fetch top 2 pages
                )
                response.raise_for_status()
                search_results = response.json().get("results", [])

                if not search_results:
                    st.warning("No relevant pages found.")
                else:
                    st.success("Relevant pages retrieved!")
                    
                    pil_images = []
                    
                    # 2. Display Source Attribution (Assignment Requirement)
                    st.subheader("📑 Source Documents")
                    cols = st.columns(len(search_results))
                    
                    for idx, res in enumerate(search_results):
                        # Decode base64 string back into an image
                        img_data = base64.b64decode(res["image_base64"])
                        img = Image.open(BytesIO(img_data))
                        pil_images.append(img)
                        
                        # Display the image in Streamlit
                        with cols[idx]:
                            st.image(
                                img, 
                                caption=f"Page {res['page_number']} (Match Score: {res['score']:.2f})", 
                                use_container_width=True
                            )

                    # 3. Generate Answer using Gemini
                    st.subheader("🤖 Answer")
                    with st.spinner("Gemini is analyzing the retrieved pages..."):
                        
                        # We pass the text prompt AND the raw images to Gemini!
                        prompt = f"Answer the user's question using ONLY the provided document pages. If the answer is in a table or chart, extract it carefully. Question: {query}"
                        
                        # The Gemini SDK accepts PIL Images directly in the contents list
                        contents = [prompt] + pil_images
                        
                        gemini_response = client.models.generate_content(
                            model='gemini-2.5-flash', # Flash is super fast and multimodal
                            contents=contents
                        )
                        
                        st.write(gemini_response.text)

            except Exception as e:
                st.error(f"Error connecting to backend: {str(e)}")