# Evaluation and Metrics

This guide explains how to evaluate your RAG system's performance using various metrics and techniques.

## Overview

We'll cover:
1. Retrieval evaluation
2. Answer quality assessment
3. System performance metrics
4. End-to-end testing

## Implementation

### 1. Retrieval Evaluation

```python
import numpy as np
from sklearn.metrics import precision_recall_curve
from typing import List, Dict

def evaluate_retrieval(retriever, test_queries: List[Dict]):
    """
    Evaluate retrieval performance.
    
    Args:
        retriever: Retriever instance
        test_queries: List of dicts with 'query' and 'relevant_docs'
        
    Returns:
        dict: Metrics including recall, precision, MRR
    """
    metrics = {
        'recall@k': [],
        'precision@k': [],
        'mrr': [],
        'ndcg': []
    }
    
    for query in test_queries:
        # Get retrieval results
        results = retriever.retrieve(
            query['query'],
            k=10
        )
        
        # Get retrieved doc ids
        retrieved_ids = [r['metadata']['id'] 
                        for r in results]
        
        # Calculate metrics
        metrics['recall@k'].append(
            calculate_recall_at_k(
                retrieved_ids,
                query['relevant_docs']
            )
        )
        
        metrics['precision@k'].append(
            calculate_precision_at_k(
                retrieved_ids,
                query['relevant_docs']
            )
        )
        
        metrics['mrr'].append(
            calculate_mrr(
                retrieved_ids,
                query['relevant_docs']
            )
        )
        
        metrics['ndcg'].append(
            calculate_ndcg(
                retrieved_ids,
                query['relevant_docs']
            )
        )
    
    # Average metrics
    return {k: np.mean(v) for k, v in metrics.items()}
```

### 2. Answer Quality Metrics

```python
from rouge_score import rouge_scorer
from bert_score import score

def evaluate_answer_quality(
    generated_answers: List[str],
    reference_answers: List[str]
):
    """
    Evaluate answer quality using various metrics.
    """
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    
    metrics = {
        'rouge': {},
        'bert_score': {},
        'answer_length': []
    }
    
    # Calculate ROUGE scores
    for gen, ref in zip(generated_answers, reference_answers):
        scores = scorer.score(gen, ref)
        
        for key, value in scores.items():
            if key not in metrics['rouge']:
                metrics['rouge'][key] = []
            metrics['rouge'][key].append(value.fmeasure)
        
        # Calculate BERTScore
        P, R, F1 = score(
            [gen],
            [ref],
            lang='en',
            verbose=False
        )
        metrics['bert_score']['precision'] = P.mean().item()
        metrics['bert_score']['recall'] = R.mean().item()
        metrics['bert_score']['f1'] = F1.mean().item()
        
        # Answer length
        metrics['answer_length'].append(len(gen.split()))
    
    # Average metrics
    for key in metrics['rouge']:
        metrics['rouge'][key] = np.mean(metrics['rouge'][key])
    
    metrics['answer_length'] = np.mean(metrics['answer_length'])
    
    return metrics
```

### 3. Performance Metrics

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class PerformanceMetrics:
    retrieval_time: float
    generation_time: float
    total_time: float
    chunks_retrieved: int
    tokens_generated: Optional[int] = None

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
        
    def measure_pipeline(self, pipeline, query):
        """Measure pipeline performance."""
        start_time = time.time()
        
        # Measure retrieval
        retrieval_start = time.time()
        contexts = pipeline.retrieve(query)
        retrieval_time = time.time() - retrieval_start
        
        # Measure generation
        generation_start = time.time()
        response = pipeline.generate(query, contexts)
        generation_time = time.time() - generation_start
        
        # Calculate metrics
        total_time = time.time() - start_time
        
        metrics = PerformanceMetrics(
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            chunks_retrieved=len(contexts)
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def get_summary(self):
        """Get performance summary."""
        if not self.metrics:
            return {}
            
        return {
            'avg_retrieval_time': np.mean(
                [m.retrieval_time for m in self.metrics]
            ),
            'avg_generation_time': np.mean(
                [m.generation_time for m in self.metrics]
            ),
            'avg_total_time': np.mean(
                [m.total_time for m in self.metrics]
            ),
            'avg_chunks': np.mean(
                [m.chunks_retrieved for m in self.metrics]
            )
        }
```

### 4. End-to-End Testing

```python
def run_e2e_evaluation(pipeline, test_cases):
    """
    Run end-to-end evaluation.
    
    Args:
        pipeline: RAG pipeline instance
        test_cases: List of test cases
        
    Returns:
        dict: Evaluation results
    """
    results = {
        'retrieval': [],
        'answer_quality': [],
        'performance': []
    }
    
    monitor = PerformanceMonitor()
    
    for test in test_cases:
        # Measure performance
        metrics = monitor.measure_pipeline(
            pipeline,
            test['query']
        )
        
        # Get pipeline response
        response = pipeline.answer_question(
            test['query']
        )
        
        # Evaluate retrieval
        retrieval_metrics = evaluate_retrieval(
            pipeline.retriever,
            [test]
        )
        
        # Evaluate answer quality
        quality_metrics = evaluate_answer_quality(
            [response['answer']],
            [test['reference_answer']]
        )
        
        # Store results
        results['retrieval'].append(retrieval_metrics)
        results['answer_quality'].append(quality_metrics)
        results['performance'].append(metrics)
    
    # Aggregate results
    summary = {
        'retrieval': aggregate_metrics(results['retrieval']),
        'answer_quality': aggregate_metrics(
            results['answer_quality']
        ),
        'performance': monitor.get_summary()
    }
    
    return summary, results
```

## Example Usage

### 1. Create Test Cases

```python
test_cases = [
    {
        'query': 'What is RAG?',
        'relevant_docs': ['doc1', 'doc2'],
        'reference_answer': 'RAG is a technique that...'
    },
    {
        'query': 'How does RAG work?',
        'relevant_docs': ['doc2', 'doc3'],
        'reference_answer': 'RAG works by retrieving...'
    }
]
```

### 2. Run Evaluation

```python
# Initialize pipeline
pipeline = RAGPipeline()

# Run evaluation
summary, detailed = run_e2e_evaluation(
    pipeline,
    test_cases
)

# Print results
print("Evaluation Summary:")
print("\nRetrieval Metrics:")
for k, v in summary['retrieval'].items():
    print(f"{k}: {v:.3f}")

print("\nAnswer Quality:")
for k, v in summary['answer_quality'].items():
    print(f"{k}: {v:.3f}")

print("\nPerformance:")
for k, v in summary['performance'].items():
    print(f"{k}: {v:.3f}s")
```

## Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(results):
    """Plot evaluation metrics."""
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Retrieval metrics
    retrieval_df = pd.DataFrame(results['retrieval'])
    sns.boxplot(data=retrieval_df, ax=axes[0,0])
    axes[0,0].set_title('Retrieval Metrics')
    
    # Answer quality
    quality_df = pd.DataFrame(results['answer_quality'])
    sns.boxplot(data=quality_df, ax=axes[0,1])
    axes[0,1].set_title('Answer Quality')
    
    # Performance
    perf_df = pd.DataFrame(
        [vars(m) for m in results['performance']]
    )
    sns.boxplot(data=perf_df, ax=axes[1,0])
    axes[1,0].set_title('Performance Metrics')
    
    plt.tight_layout()
    return fig
```

## Best Practices

1. **Test Data Preparation**
   - Create diverse test cases
   - Include edge cases
   - Maintain gold standard answers

2. **Evaluation Strategy**
   - Regular benchmarking
   - A/B testing new features
   - Monitoring drift

3. **Metric Selection**
   - Choose task-appropriate metrics
   - Consider human evaluation
   - Track trends over time

## Next Steps

- Implement [Custom Knowledge Base](custom-kb.md)
- Explore [Deployment Options](../deployment/local.md)
- Setup [Monitoring](../deployment/monitoring.md)