# Vector Store

This section explains how we use FAISS (Facebook AI Similarity Search) to store and retrieve embeddings efficiently.

## Overview

FAISS is a library for efficient similarity search of dense vectors. We use it to:
1. Store document embeddings
2. Perform fast nearest neighbor search
3. Scale to large document collections

## Implementation

### Setting up FAISS

```python
import faiss
import numpy as np

def create_faiss_index(dimension, index_type='L2'):
    """
    Create a FAISS index.
    
    Args:
        dimension (int): Embedding dimension
        index_type (str): Type of index ('L2' or 'IP')
        
    Returns:
        faiss.Index: Initialized FAISS index
    """
    if index_type == 'L2':
        return faiss.IndexFlatL2(dimension)
    elif index_type == 'IP':
        return faiss.IndexFlatIP(dimension)
    else:
        raise ValueError("Invalid index type")
```

### Adding Vectors

```python
def add_to_index(index, embeddings):
    """
    Add embeddings to the FAISS index.
    
    Args:
        index (faiss.Index): FAISS index
        embeddings (np.ndarray): Matrix of embeddings
        
    Returns:
        int: Number of vectors added
    """
    # Convert to float32 (required by FAISS)
    embeddings = np.array(embeddings).astype('float32')
    
    # Add to index
    index.add(embeddings)
    
    return index.ntotal
```

### Similarity Search

```python
def search_index(index, query_embedding, k=5):
    """
    Search for similar vectors.
    
    Args:
        index (faiss.Index): FAISS index
        query_embedding (np.ndarray): Query vector
        k (int): Number of results to return
        
    Returns:
        tuple: (distances, indices)
    """
    # Ensure query is float32 and 2D
    query_embedding = np.array(query_embedding).astype('float32')
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Perform search
    distances, indices = index.search(query_embedding, k)
    
    return distances[0], indices[0]
```

## Advanced Index Types

### IVF Index (for Large Datasets)

```python
def create_ivf_index(dimension, nlist=100):
    """
    Create an IVF index for faster search.
    
    Args:
        dimension (int): Embedding dimension
        nlist (int): Number of clusters
        
    Returns:
        faiss.Index: IVF index
    """
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(
        quantizer, 
        dimension,
        nlist,
        faiss.METRIC_L2
    )
    
    return index
```

### GPU Acceleration

```python
def use_gpu_index(index):
    """
    Move index to GPU.
    
    Args:
        index (faiss.Index): CPU index
        
    Returns:
        faiss.Index: GPU index
    """
    res = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(res, 0, index)
```

## Complete Example

```python
def setup_vector_store(embeddings, dimension=384):
    """
    Set up a complete vector store.
    
    Args:
        embeddings (np.ndarray): Initial embeddings
        dimension (int): Embedding dimension
        
    Returns:
        tuple: (index, total_vectors)
    """
    # Create index
    index = create_faiss_index(dimension)
    
    # Add vectors
    total = add_to_index(index, embeddings)
    
    print(f"Added {total} vectors to index")
    
    return index, total

# Example usage
dimension = 384  # for all-MiniLM-L6-v2
embeddings = np.random.random((100, dimension)).astype('float32')

# Setup store
index, total = setup_vector_store(embeddings)

# Search example
query = np.random.random(dimension).astype('float32')
distances, indices = search_index(index, query, k=5)

print("Search results:")
for d, i in zip(distances, indices):
    print(f"Index: {i}, Distance: {d:.4f}")
```

## Performance Optimization

### 1. Index Types
Choose based on your needs:
- **Flat**: Best accuracy, slower
- **IVF**: Faster search, slight accuracy loss
- **HNSW**: Very fast, memory intensive

### 2. GPU Usage
When to use GPU:
- Large datasets (>1M vectors)
- Need for real-time search
- GPU memory available

### 3. Batch Processing
For multiple queries:

```python
def batch_search(index, queries, k=5):
    """
    Search for multiple queries at once.
    
    Args:
        index (faiss.Index): FAISS index
        queries (np.ndarray): Query vectors
        k (int): Results per query
        
    Returns:
        tuple: (distances, indices)
    """
    queries = np.array(queries).astype('float32')
    return index.search(queries, k)
```

## Storage and Persistence

### Save Index

```python
def save_index(index, filepath):
    """
    Save FAISS index to disk.
    
    Args:
        index (faiss.Index): FAISS index
        filepath (str): Output path
    """
    faiss.write_index(index, filepath)
```

### Load Index

```python
def load_index(filepath):
    """
    Load FAISS index from disk.
    
    Args:
        filepath (str): Path to saved index
        
    Returns:
        faiss.Index: Loaded index
    """
    return faiss.read_index(filepath)
```

## Best Practices

1. **Index Selection**
    - Use Flat index for <1M vectors
    - Use IVF for >1M vectors
    - Consider HNSW for speed priority

2. **Resource Management**
    - Monitor memory usage
    - Use GPU strategically
    - Implement batch processing

3. **Maintenance**
    - Regular index saves
    - Monitoring index size
    - Performance testing

## Next Steps

- Learn about [Retrieval](retriever.md)
- Explore [End-to-End Pipeline](pipeline.md)
- See [Examples](../examples/basic-retrieval.md)