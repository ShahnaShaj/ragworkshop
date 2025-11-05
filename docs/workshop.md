# Workshop Implementation

This page provides a detailed walkthrough of implementing the RAG system.

## 1. Configure API Key

First, you'll need to configure your Gemini API key:

```python
import os
import getpass
import google.generativeai as genai

# Configure the Gemini API
os.environ["GEMINI_API_KEY"] = "your-api-key-here"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize the generative model
model = genai.GenerativeModel('gemini-2.5-flash')
```

## 2. Process PDF Documents

The system can process PDF documents to create a knowledge base:

```python
from pypdf import PdfReader

def process_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
    
    # Split the text into chunks
    text_chunks = [pdf_text[i:i + 1000] for i in range(0, len(pdf_text), 1000)]
    return text_chunks
```

## 3. Create Text Embeddings

We use Sentence Transformers to create embeddings:

```python
from sentence_transformers import SentenceTransformer

# Load a pre-trained model from Hugging Face
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for the text chunks
embeddings = embedding_model.encode(text_chunks)
```

## 4. Build Vector Store

FAISS is used to create a vector store for efficient similarity search:

```python
import faiss
import numpy as np

# Create a FAISS index
d = embeddings.shape[1]  # dimension of the embeddings
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings).astype('float32'))
```

## 5. RAG Implementation

The core RAG functionality:

```python
def get_rag_answer(query, k=3):
    """
    Performs retrieval-augmented generation.
    
    Args:
        query: The user's question
        k: Number of relevant chunks to retrieve
    """
    # 1. Retrieval
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)

    # Get relevant text chunks
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    
    # 2. Augmentation
    context = "\n\n".join(retrieved_chunks)
    augmented_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

    # 3. Generation
    response = model.generate_content(augmented_prompt)
    return response.text
```

## 6. Interactive Interface

The system includes a Gradio-based interface for easy interaction:

```python
import gradio as gr

def chat_with_pdf(question):
    if not question.strip():
        return "Please enter a question."
    
    try:
        answer = get_rag_answer(question)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=chat_with_pdf,
    inputs=gr.Textbox(lines=3, label="Your Question"),
    outputs=gr.Textbox(lines=5, label="Answer")
)