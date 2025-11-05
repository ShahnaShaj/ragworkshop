# RAG Workshop Implementation Walkthrough

This document provides a detailed walkthrough of the RAG system implementation from the notebook.

## 1. Dependencies Installation

The notebook implements a RAG system that combines Google's Gemini model with local document processing. The required dependencies are:

```python
!pip install -q google-generativeai pypdf sentence-transformers faiss-cpu
```

Each library serves a specific purpose:
- `google-generativeai`: Interface with Gemini model
- `pypdf`: PDF document processing
- `sentence-transformers`: Text embedding generation
- `faiss-cpu`: Vector similarity search
- `gradio`: Web interface creation

## 2. API Configuration

```python
import os
import getpass
import google.generativeai as genai

# Get API key securely
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")

# Configure Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash')
```

### Key Points:
- Uses environment variables for API key
- Secure input with getpass
- Initializes Gemini model
- Uses 'gemini-2.5-flash' for faster responses

## 3. Document Processing

```python
from google.colab import files
from pypdf import PdfReader

def process_pdf(file):
    """Process uploaded PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_chunks(text, chunk_size=1000):
    """Split text into chunks."""
    return [text[i:i + chunk_size] 
            for i in range(0, len(text), chunk_size)]
```

### Implementation Details:
- Uses PdfReader for text extraction
- Simple chunking strategy (1000 characters)
- No overlap between chunks
- Handles PDF upload through Colab interface

### Improvements Possible:
- Add chunk overlap
- Implement smarter chunking (sentence boundaries)
- Add metadata tracking
- Handle different file types

## 4. Embedding Generation

```python
from sentence_transformers import SentenceTransformer

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    """Generate embeddings for text chunks."""
    return embedding_model.encode(chunks)
```

### Model Choice:
- all-MiniLM-L6-v2: Good balance of speed and quality
- 384-dimensional embeddings
- Optimized for semantic similarity

### Process Flow:
1. Load pre-trained model
2. Convert text chunks to embeddings
3. Return numpy array of embeddings

## 5. Vector Store Setup

```python
import faiss
import numpy as np

def create_vector_store(embeddings):
    """Create FAISS index for embeddings."""
    # Get embedding dimension
    dimension = embeddings.shape[1]
    
    # Create L2 index
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors
    index.add(np.array(embeddings).astype('float32'))
    
    return index
```

### FAISS Configuration:
- Uses L2 distance metric
- Flat index for exact search
- float32 data type for vectors
- No quantization or clustering

### Design Choices:
- Simple IndexFlatL2 for small datasets
- Could use IVF or HNSW for scaling
- Direct memory storage (no persistence)

## 6. RAG Implementation

```python
def get_rag_answer(query, k=3):
    """
    Performs retrieval-augmented generation.
    
    Args:
        query: User question
        k: Number of chunks to retrieve
        
    Returns:
        Generated answer
    """
    # 1. Retrieval
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(
        np.array(query_embedding).astype('float32'),
        k
    )
    
    # 2. Augmentation
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)
    
    # 3. Prompt Construction
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    # 4. Generation
    response = model.generate_content(prompt)
    return response.text
```

### Function Components:

1. **Query Processing**:
   - Converts query to embedding
   - Same model as document embeddings
   - Single vector output

2. **Retrieval**:
   - Uses FAISS k-nearest neighbors
   - Returns top-k most similar chunks
   - Includes distance scores

3. **Context Assembly**:
   - Combines retrieved chunks
   - Simple concatenation with newlines
   - No context ordering/ranking

4. **Prompt Engineering**:
   - Clear instruction format
   - Explicit context separation
   - Simple question-answer structure

5. **Generation**:
   - Uses Gemini model
   - No temperature control
   - No response formatting

### Design Considerations:
- k=3 balances context size and relevance
- Simple prompt template for reliability
- No streaming or async processing
- No error handling for long inputs

## 7. User Interface

```python
import gradio as gr

def chat_with_pdf(question, history):
    """Handle chat interface."""
    if not question.strip():
        return "Please enter a question."
        
    try:
        return get_rag_answer(question)
    except Exception as e:
        return f"Error: {str(e)}"

# Create interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Header
    gr.Markdown("# 📚 RAG-Powered PDF Q&A System")
    
    # Input/Output
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Your Question",
                lines=3
            )
            submit_btn = gr.Button("Ask")
            
        with gr.Column():
            answer_output = gr.Textbox(
                label="Answer",
                lines=10
            )
    
    # Example questions
    gr.Examples([
        ["What is the main topic?"],
        ["Summarize key points"],
        ["What details are mentioned?"]
    ], inputs=question_input)
    
    # Event handlers
    submit_btn.click(
        fn=lambda q: chat_with_pdf(q, None),
        inputs=question_input,
        outputs=answer_output
    )
```

### UI Components:

1. **Layout**:
    - Two-column design
    - Clean, modern theme
    - Responsive layout

2. **Input Features**:
    - Multi-line question input
    - Submit button
    - Example questions

3. **Output Display**:
    - Multi-line answer box
    - Error handling
    - Clear formatting

4. **Interactions**:
    - Click and submit events
    - Simple callback structure
    - No state management

### Implementation Notes:
- Uses Gradio Blocks for flexibility
- No conversation history
- Basic error handling
- Example questions for guidance

## 8. Current Implementation Features

1. **Document Processing**:
    - Full PDF text extraction
    - Fixed-size chunk splitting (1000 chars)
    - Sequential page processing

2. **Retrieval System**:
    - Exact vector similarity search
    - Top-3 chunk retrieval
    - L2 distance metric

3. **Question Answering**:
    - Context-aware responses
    - Basic error handling
    - Example questions provided

4. **User Interface**:
    - Two-column layout
    - Responsive design
    - Input validation
    - Debug mode enabled