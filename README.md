# Multi-Modal Document QA System

This repository contains a Retrieval-Augmented Generation (RAG) system for complex documents with text, tables, and images. It uses visual embeddings and a multi-modal language model to produce answers grounded in the uploaded document context.

## Architecture

The system is divided into three main components:

* **Backend (Ingestion and Retrieval):** A FastAPI server running in a GPU environment. It uses the ColPali Vision-Language Model to convert PDF pages into multi-vector visual embeddings.
* **Vector Database:** Qdrant stores multi-vector embeddings and base64-encoded page images. It uses a MAX_SIM comparator for similarity search.
* **Frontend (UI and Generation):** A Streamlit app that fetches relevant visual pages from Qdrant and sends raw images to Gemini 2.5 Flash for answer generation.

## Key Features

* **Visual Retrieval:** Encodes full-page visual structure, preserving layout details such as tables and charts.
* **Multi-Modal Generation:** Sends raw images to Gemini 2.5 Flash so it can read visual content directly.
* **Sequential Processing:** Ingests documents page-by-page to reduce GPU memory pressure.
* **Source Attribution:** Displays retrieved pages alongside generated answers for transparency.

## Prerequisites

* Python 3.14 or higher
* Access to a GPU environment (for example, Google Colab) for backend ingestion
* Qdrant Cloud account and API key
* Google Gemini API key
* ngrok account and auth token

## Setup and Installation

### 1. Backend Setup (GPU Environment)

1. Open `notebooks/colpali_server.ipynb` in your GPU environment.
2. Add your Qdrant API key and ngrok auth token where indicated.
3. Run all notebook cells to start the FastAPI backend.
4. Copy the generated ngrok public URL.

### 2. Frontend Setup (Local Machine)

1. Clone this repository.
2. Create a `.env` file in the project root:

```text
COLAB_API_URL=<your_ngrok_public_url>
GEMINI_API_KEY=<your_gemini_api_key>
```

3. Install dependencies:

```bash
pip install -e .
```

4. Start the Streamlit app:

```bash
streamlit run src/app.py
```

## Usage

1. Open the Streamlit app in your browser.
2. Upload a PDF in the sidebar and click **Process Document**.
3. Enter a query in the chat input.
4. Review the generated answer and retrieved source pages.

## Evaluation

Run the evaluation script:

```bash
python src/evaluate.py
```

This script benchmarks retrieval and generation on text, table, and chart/image-style queries. Results are saved to `evaluation_results.json`.