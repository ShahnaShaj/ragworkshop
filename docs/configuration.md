# Configuration Guide

This guide explains all the configurable parameters and settings in the RAG Workshop project.

## Environment Variables

Create a `.env` file in your project root with the following settings:

```bash
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
TOP_K=3
```

## Configuration Parameters

### Document Processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Number of characters per text chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `CLEAN_TEXT` | True | Whether to clean extracted text |

### Embedding Generation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | 'all-MiniLM-L6-v2' | SentenceTransformer model name |
| `EMBEDDING_DIMENSION` | 384 | Embedding vector size |
| `NORMALIZE_EMBEDDINGS` | True | Whether to normalize vectors |

### Vector Store

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INDEX_TYPE` | 'L2' | FAISS index type (L2 or IP) |
| `NLIST` | 100 | Number of cells for IVF index |
| `NPROBE` | 10 | Number of cells to probe |

### Retrieval

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_K` | 3 | Number of chunks to retrieve |
| `MIN_SIMILARITY` | 0.6 | Minimum similarity threshold |
| `MAX_CONTEXT_LENGTH` | 2000 | Maximum context window size |

### LLM Generation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | 'gemini-2.5-flash' | Gemini model version |
| `TEMPERATURE` | 0.7 | Generation temperature |
| `MAX_TOKENS` | 1000 | Maximum response length |

## Custom Configuration

You can create a `config.yaml` file for custom settings:

```yaml
document_processing:
  chunk_size: 1000
  chunk_overlap: 200
  clean_text: true

embedding:
  model_name: all-MiniLM-L6-v2
  dimension: 384
  normalize: true

vector_store:
  index_type: L2
  nlist: 100
  nprobe: 10

retrieval:
  top_k: 3
  min_similarity: 0.6
  max_context_length: 2000

llm:
  model_name: gemini-2.5-flash
  temperature: 0.7
  max_tokens: 1000
```

## Loading Configuration

```python
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
```

## Environment Setup

1. Copy the `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your API keys and settings

3. Load environment variables in your code:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

## Configuration Best Practices

1. **API Keys**
      - Never commit API keys to version control
      - Use environment variables for sensitive data
      - Rotate keys regularly

2. **Model Settings**
      - Start with default values
      - Adjust based on your specific use case
      - Monitor performance metrics

3. **Resource Usage**
      - Balance chunk size with memory usage
      - Adjust vector store parameters for scale
      - Optimize context window size

4. **Performance Tuning**
      - Monitor retrieval accuracy
      - Adjust similarity thresholds
      - Fine-tune generation parameters