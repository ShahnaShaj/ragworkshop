# Retriever Implementation

The retriever component is responsible for finding and returning the most relevant text chunks for a given query. This guide explains how our retrieval system works and how to optimize it.

## Overview

The retriever:
1. Takes a user query
2. Converts it to an embedding
3. Finds similar document chunks
4. Returns the most relevant contexts

## Core Implementation

### Basic Retriever

```python
def retrieve_context(query, embedding_model, index, text_chunks, k=3):
    """
    Retrieve relevant context for a query.
    
    Args:
        query (str): User question
        embedding_model: Initialized SentenceTransformer
        index: FAISS index
        text_chunks (list): Original text chunks
        k (int): Number of chunks to retrieve
        
    Returns:
        list: Retrieved text chunks
        list: Distance scores
    """
    # Create query embedding
    query_embedding = embedding_model.encode([query])
    
    # Search the index
    distances, indices = index.search(
        np.array(query_embedding).astype('float32'),
        k
    )
    
    # Get the text chunks
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    
    return retrieved_chunks, distances[0]
```

### Advanced Retriever with Metadata

```python
class Retriever:
    def __init__(self, embedding_model, index, chunks, metadata=None):
        self.embedding_model = embedding_model
        self.index = index
        self.chunks = chunks
        self.metadata = metadata or {}
        
    def retrieve(self, query, k=3, threshold=0.6):
        """
        Enhanced retrieval with metadata and filtering.
        """
        # Get embeddings
        query_emb = self.embedding_model.encode([query])
        
        # Search
        D, I = self.index.search(
            np.array(query_emb).astype('float32'),
            k
        )
        
        # Filter by threshold
        mask = D[0] < threshold
        indices = I[0][mask]
        scores = D[0][mask]
        
        # Get chunks and metadata
        results = []
        for idx, score in zip(indices, scores):
            results.append({
                'text': self.chunks[idx],
                'metadata': self.metadata.get(idx, {}),
                'score': float(score)
            })
            
        return results
```

## Reranking

For better precision, implement a reranking step:

```python
from sentence_transformers import CrossEncoder

class RerankedRetriever(Retriever):
    def __init__(self, *args, reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        super().__init__(*args)
        self.reranker = CrossEncoder(reranker_model)
        
    def retrieve(self, query, k=10, final_k=3):
        # Get initial candidates
        candidates = super().retrieve(query, k=k)
        
        # Rerank
        pairs = [[query, c['text']] for c in candidates]
        scores = self.reranker.predict(pairs)
        
        # Sort and filter
        ranked = sorted(zip(candidates, scores), 
                       key=lambda x: x[1], 
                       reverse=True)
        
        return [r[0] for r in ranked[:final_k]]
```

## Hybrid Retrieval

Combine dense and sparse retrieval:

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, dense_retriever, chunks):
        self.dense = dense_retriever
        # Initialize BM25
        tokenized_chunks = [chunk.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        self.chunks = chunks
        
    def retrieve(self, query, k=3, alpha=0.5):
        # Dense scores
        dense_results = self.dense.retrieve(query, k=k)
        dense_scores = {r['text']: r['score'] 
                       for r in dense_results}
        
        # Sparse scores
        sparse_scores = self.bm25.get_scores(query.split())
        norm_sparse = (sparse_scores - np.min(sparse_scores)) / \
                     (np.max(sparse_scores) - np.min(sparse_scores))
        
        # Combine scores
        final_scores = {}
        for i, chunk in enumerate(self.chunks):
            if chunk in dense_scores:
                final_scores[chunk] = alpha * dense_scores[chunk] + \
                                    (1-alpha) * norm_sparse[i]
        
        # Sort and return top k
        ranked = sorted(final_scores.items(), 
                       key=lambda x: x[1], 
                       reverse=True)
        
        return [{'text': r[0], 'score': r[1]} 
                for r in ranked[:k]]
```

## Configuration

Retriever parameters in `config.yaml`:

```yaml
retriever:
  initial_k: 10
  final_k: 3
  similarity_threshold: 0.6
  hybrid:
    enabled: true
    alpha: 0.5
  reranking:
    enabled: true
    model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Best Practices

1. **Parameter Tuning**
    - Start with larger `k` and filter down
    - Adjust threshold based on corpus
    - Monitor retrieval quality

2. **Performance**
    - Cache query embeddings
    - Batch similar queries
    - Use approximate search for scale

3. **Quality**
    - Implement reranking for precision
    - Use hybrid retrieval for robustness
    - Track retrieval metrics

## Evaluation

### Retrieval Metrics

```python
def evaluate_retrieval(retriever, queries, ground_truth):
    """
    Evaluate retriever performance.
    
    Args:
        retriever: Retriever instance
        queries: List of test queries
        ground_truth: Dict mapping queries to relevant chunks
        
    Returns:
        dict: Metrics including recall, precision, MRR
    """
    metrics = {
        'recall': [],
        'precision': [],
        'mrr': []
    }
    
    for query in queries:
        results = retriever.retrieve(query)
        retrieved = {r['text'] for r in results}
        relevant = ground_truth[query]
        
        # Calculate metrics
        recall = len(retrieved & relevant) / len(relevant)
        precision = len(retrieved & relevant) / len(retrieved)
        
        # MRR calculation
        ranks = []
        for i, r in enumerate(results, 1):
            if r['text'] in relevant:
                ranks.append(1/i)
        mrr = max(ranks) if ranks else 0
        
        metrics['recall'].append(recall)
        metrics['precision'].append(precision)
        metrics['mrr'].append(mrr)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

## Debugging

Common issues and solutions:

1. **Poor Retrieval Quality**
   - Check embedding quality
   - Validate chunk sizes
   - Adjust similarity threshold

2. **Slow Retrieval**
   - Profile index search time
   - Consider approximate search
   - Optimize batch size

3. **Out of Memory**
   - Reduce initial k
   - Use memory-efficient index
   - Stream results if needed

## Next Steps

- Learn about [Generator Integration](generator.md)
- Explore the [End-to-End Pipeline](pipeline.md)
- See [Example Queries](../examples/basic-retrieval.md)