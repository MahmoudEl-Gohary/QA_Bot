import os
import requests
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from google import genai
import json

load_dotenv()
COLAB_API_URL = os.getenv("COLAB_API_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

BENCHMARK_QUERIES = [
    {"type": "Text Retrieval", "question": "What is the main topic or abstract of this document?"},
    {"type": "Table Extraction", "question": "Extract the specific data from the first table you see in the document."},
    {"type": "Chart/Image Analysis", "question": "Describe the main trend shown in the primary chart or figure of this document."}
]

def run_evaluation():
    results = []
    print("Starting Multi-Modal Benchmark Evaluation...\n")
    
    for item in BENCHMARK_QUERIES:
        query_type = item["type"]
        question = item["question"]
        print(f"[{query_type}] Testing Question: {question}")
        
        try:
            # 1. Search Qdrant via Colab
            resp = requests.post(
                f"{COLAB_API_URL}/search",
                json={"query": question, "top_k": 1} # get the best page
            )
            resp.raise_for_status()
            search_res = resp.json().get("results", [])[0]
            
            print(f"   -> Retrieved Page {search_res['page_number']} (Score: {search_res['score']:.3f})")
            
            # 2. Prepare Image for Gemini
            img_data = base64.b64decode(search_res["image_base64"])
            img = Image.open(BytesIO(img_data))
            
            # 3. Generate Answer
            prompt = f"Answer this using ONLY the provided document image: {question}"
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, img]
            )
            
            answer = response.text.strip()
            print(f"   -> Answer: {answer[:100]}...\n")
            
            results.append({
                "type": query_type,
                "question": question,
                "retrieved_page": search_res['page_number'],
                "retrieval_score": search_res['score'],
                "generated_answer": answer
            })
            
        except Exception as e:
            print(f"   -> Error: {str(e)}\n")

    # Save to a JSON file for the report
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Evaluation complete. Results saved to evaluation_results.json")

if __name__ == "__main__":
    run_evaluation()