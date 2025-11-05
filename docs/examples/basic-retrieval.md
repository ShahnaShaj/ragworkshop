# Basic Text Retrieval Example

This guide walks through a complete example of setting up and using the RAG system for basic text retrieval and question answering.

## Setup

First, let's set up the basic environment and import required packages:

```python
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader

# Configure API
os.environ["GEMINI_API_KEY"] = "your-api-key-here"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
```

## Sample Document

Let's create a simple example document:

```python
sample_text = """
Retrieval-Augmented Generation (RAG) is a technique that enhances language models
by providing them with relevant external information during generation. This approach
combines the benefits of retrieval-based and generation-based methods.

RAG works in three main steps:
1. When a user asks a question, the system searches a knowledge base to find relevant information
2. The retrieved information is combined with the original question
3. This enhanced prompt is sent to a language model to generate an answer

Key benefits of RAG include:
- Improved accuracy by grounding responses in specific documents
- Reduced hallucination by providing concrete context
- Ability to answer questions about new or updated information
- More transparent and verifiable responses
"""

# Split into chunks
chunks = [sample_text[i:i + 200] for i in range(0, len(sample_text), 200)]
```

## Creating Embeddings

Initialize the embedding model and create embeddings:

```python
# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = embedding_model.encode(chunks)
print(f"Created {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
```

## Setting up FAISS

Create and populate the vector store:

```python
# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add vectors to index
index.add(np.array(embeddings).astype('float32'))
print(f"Added {index.ntotal} vectors to index")
```

## Basic Retrieval Function

```python
def retrieve_context(query, k=2):
    """
    Retrieve relevant chunks for a query.
    
    Args:
        query (str): User question
        k (int): Number of chunks to retrieve
        
    Returns:
        list: Retrieved text chunks
        list: Similarity scores
    """
    # Create query embedding
    query_emb = embedding_model.encode([query])
    
    # Search index
    distances, indices = index.search(
        np.array(query_emb).astype('float32'),
        k
    )
    
    # Get chunks and scores
    results = []
    for idx, score in zip(indices[0], distances[0]):
        results.append({
            'text': chunks[idx],
            'score': float(score)
        })
    
    return results
```

## Question Answering

Let's implement a simple question-answering function:

```python
def answer_question(query):
    """
    Answer a question using RAG.
    """
    # Get relevant chunks
    results = retrieve_context(query, k=2)
    
    # Build context
    context = "\n\n".join(r['text'] for r in results)
    
    # Create prompt
    prompt = f"""Based on the following context, answer the question.
    If you cannot find relevant information, say so.

    Context:
    {context}

    Question: {query}

    Answer:"""
    
    # Generate answer
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    return {
        'answer': response.text,
        'sources': results
    }
```

## Example Usage

Let's try some example questions:

```python
# Example 1: Basic factual question
question1 = "What is RAG and how does it work?"
result1 = answer_question(question1)
print("Question:", question1)
print("Answer:", result1['answer'])
print("\nSources:")
for src in result1['sources']:
    print(f"Score: {src['score']:.3f}")
    print(f"Text: {src['text'][:100]}...")
print("\n")

# Example 2: Specific detail question
question2 = "What are the benefits of using RAG?"
result2 = answer_question(question2)
print("Question:", question2)
print("Answer:", result2['answer'])
print("\nSources:")
for src in result2['sources']:
    print(f"Score: {src['score']:.3f}")
    print(f"Text: {src['text'][:100]}...")
```

## Expected Output

For the first question, you might see:

```
Question: What is RAG and how does it work?
Answer: RAG (Retrieval-Augmented Generation) is a technique that enhances language models by providing them with external information during generation. It works in three main steps:

1. When a user asks a question, the system searches a knowledge base for relevant information
2. The retrieved information is combined with the original question
3. This enhanced prompt is sent to a language model to generate an answer

Sources:
Score: 0.235
Text: Retrieval-Augmented Generation (RAG) is a technique that enhances language models by providing them with relev...
Score: 0.412
Text: RAG works in three main steps: 1. When a user asks a question, the system searches a knowledge base to find...
```

## Advanced Usage

### 1. Filtering by Score

```python
def retrieve_with_threshold(query, threshold=0.5):
    """Retrieve chunks with score filtering."""
    results = retrieve_context(query, k=5)
    return [r for r in results if r['score'] < threshold]
```

### 2. Contextual History

```python
class RAGChat:
    def __init__(self):
        self.history = []
        
    def chat(self, query):
        # Get answer
        result = answer_question(query)
        
        # Update history
        self.history.append({
            'query': query,
            'response': result
        })
        
        return result
    
    def get_chat_history(self):
        return self.history
```

## Troubleshooting

Common issues and solutions:

1. **Poor Retrieval Results**
   - Try increasing `k`
   - Adjust chunk size
   - Check embedding quality

2. **Irrelevant Answers**
   - Review context window
   - Adjust prompt template
   - Lower temperature

3. **Performance Issues**
   - Batch embeddings
   - Optimize index
   - Cache results

## Next Steps

- Try [PDF Document Processing](pdf-processing.md)
- Build a [Custom Knowledge Base](custom-kb.md)
- Learn about [Evaluation](evaluation.md)