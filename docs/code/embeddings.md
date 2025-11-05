# Embedding Generation

This section explains how we generate embeddings from text chunks using Sentence Transformers.

## Overview

Embeddings are numerical representations of text that capture semantic meaning. We use the `sentence-transformers` library to convert text chunks into dense vector embeddings.

## Implementation

### Setup

```python
from sentence_transformers import SentenceTransformer

def initialize_embedding_model(model_name='all-MiniLM-L6-v2'):
    """
    Initialize the embedding model.
    
    Args:
        model_name (str): Name of the pretrained model
        
    Returns:
        SentenceTransformer: Initialized model
    """
    return SentenceTransformer(model_name)
```

### Generating Embeddings

```python
import numpy as np

def generate_embeddings(texts, model):
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts (list): List of text chunks
        model (SentenceTransformer): Initialized model
        
    Returns:
        np.ndarray: Matrix of embeddings
    """
    # Generate embeddings
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    return embeddings
```

## Model Selection

We use the `all-MiniLM-L6-v2` model by default because it:

1. Provides good performance for most use cases
2. Has a reasonable embedding dimension (384)
3. Is fast and memory-efficient
4. Works well with semantic search

### Available Models

| Model Name | Dimension | Speed | Performance |
|------------|-----------|-------|-------------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| all-mpnet-base-v2 | 768 | Medium | Better |
| all-roberta-large-v1 | 1024 | Slow | Best |

## Usage Example

```python
# Initialize model
model_name = 'all-MiniLM-L6-v2'
embedding_model = initialize_embedding_model(model_name)

# Generate embeddings for text chunks
chunks = [
    "This is the first document.",
    "Another document with different content.",
    "A third document for demonstration."
]

embeddings = generate_embeddings(chunks, embedding_model)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {embeddings.shape[1]}")
```

## Configuration

Customize embedding generation in your config file:

```yaml
embedding:
  model_name: all-MiniLM-L6-v2
  normalize: true
  batch_size: 32
  show_progress: true
```

## Advanced Usage

### Batch Processing

For large document collections:

```python
def batch_generate_embeddings(texts, model, batch_size=32):
    """
    Generate embeddings in batches.
    
    Args:
        texts (list): List of text chunks
        model (SentenceTransformer): Initialized model
        batch_size (int): Batch size
        
    Returns:
        np.ndarray: Matrix of embeddings
    """
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)
```

### Embedding Pooling

Combine multiple embeddings:

```python
def pool_embeddings(embeddings, method='mean'):
    """
    Pool multiple embeddings into one.
    
    Args:
        embeddings (np.ndarray): Matrix of embeddings
        method (str): Pooling method ('mean' or 'max')
        
    Returns:
        np.ndarray: Pooled embedding
    """
    if method == 'mean':
        return np.mean(embeddings, axis=0)
    elif method == 'max':
        return np.max(embeddings, axis=0)
    else:
        raise ValueError("Invalid pooling method")
```

## Best Practices

1. **Model Selection**
    - Choose based on your specific needs
    - Consider speed vs. accuracy tradeoff
    - Test different models for your use case

2. **Memory Management**
    - Use batch processing for large datasets
    - Clear GPU memory when needed
    - Monitor memory usage

3. **Performance Optimization**
    - Normalize embeddings
    - Use appropriate batch sizes
    - Cache embeddings when possible

## Common Issues

1. **Out of Memory**
    - Reduce batch size
    - Use CPU if GPU memory is limited
    - Process in chunks

2. **Slow Processing**
    - Use smaller models
    - Increase batch size
    - Enable GPU acceleration

3. **Quality Issues**
    - Try different models
    - Adjust text preprocessing
    - Validate embedding quality

## Next Steps

- Learn about [Vector Store](vector-store.md) setup
- Explore [Retrieval](retriever.md) methods
- See [Example Usage](../examples/basic-retrieval.md)