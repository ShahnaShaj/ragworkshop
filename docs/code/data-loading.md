# Data Loading

The first step in our RAG pipeline is loading and processing documents. This guide explains how we handle document ingestion and preprocessing.

## PDF Document Loading

We use the `pypdf` library to load and extract text from PDF documents:

```python
from pypdf import PdfReader

def load_pdf(file_path):
    """
    Load and extract text from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    reader = PdfReader(file_path)
    pdf_text = ""
    
    # Extract text from each page
    for page in reader.pages:
        pdf_text += page.extract_text()
        
    return pdf_text
```

## Text Chunking

After extracting text, we split it into manageable chunks:

```python
def create_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into overlapping chunks.
    
    Args:
        text (str): Input text to chunk
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position with overlap
        end = start + chunk_size
        
        # Add chunk to list
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start = end - chunk_overlap
        
    return chunks
```

## Text Preprocessing

Before chunking, we clean and normalize the text:

```python
import re

def clean_text(text):
    """
    Clean and normalize text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Normalize whitespace
    text = text.strip()
    
    return text
```

## Complete Pipeline

Here's how to use all components together:

```python
def process_document(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a PDF document from start to finish.
    
    Args:
        file_path (str): Path to PDF file
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: Processed text chunks
    """
    # Load PDF
    raw_text = load_pdf(file_path)
    
    # Clean text
    cleaned_text = clean_text(raw_text)
    
    # Create chunks
    chunks = create_chunks(
        cleaned_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return chunks
```

## Usage Example

```python
# Process a PDF document
file_path = "example.pdf"
chunks = process_document(
    file_path,
    chunk_size=1000,
    chunk_overlap=200
)

print(f"Created {len(chunks)} chunks")

# Preview first chunk
print("\nFirst chunk preview:")
print(chunks[0][:200] + "...")
```

## Configuration

You can customize the processing parameters in your config file:

```yaml
document_processing:
  chunk_size: 1000
  chunk_overlap: 200
  clean_text: true
```

## Best Practices

1. **Chunk Size**
    - Experiment with different sizes
    - Consider model context limits
    - Balance detail vs. context

2. **Text Cleaning**
    - Preserve meaningful punctuation
    - Remove irrelevant characters
    - Maintain sentence structure

3. **Error Handling**
    - Validate PDF files
    - Handle encoding issues
    - Log processing errors

## Next Steps

- Learn about [Embedding Generation](embeddings.md)
- Explore [Vector Store](vector-store.md) setup
- See [Example Usage](../examples/pdf-processing.md)