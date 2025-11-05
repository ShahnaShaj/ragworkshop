# PDF Document Processing

This guide demonstrates how to process PDF documents effectively in your RAG system.

## Overview

Processing PDFs involves:
1. Text extraction
2. Layout analysis
3. Content cleaning
4. Intelligent chunking
5. Metadata preservation

## Implementation

### 1. Enhanced PDF Processing

```python
from pypdf import PdfReader
import re
from typing import List, Dict

class PDFProcessor:
    def __init__(self, config=None):
        self.config = config or self.default_config()
        
    @staticmethod
    def default_config():
        return {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'min_chunk_size': 100,
            'clean_text': True,
            'preserve_layout': True
        }
        
    def process_pdf(self, file_path: str) -> List[Dict]:
        """
        Process PDF and return chunks with metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            list: List of dicts with text and metadata
        """
        # Read PDF
        reader = PdfReader(file_path)
        
        # Extract text and metadata
        chunks = []
        for page_num, page in enumerate(reader.pages, 1):
            # Get page text
            text = page.extract_text()
            
            # Clean text if enabled
            if self.config['clean_text']:
                text = self.clean_text(text)
            
            # Create chunks
            page_chunks = self.create_chunks(
                text,
                self.config['chunk_size'],
                self.config['chunk_overlap']
            )
            
            # Add metadata
            for i, chunk in enumerate(page_chunks):
                chunks.append({
                    'text': chunk,
                    'metadata': {
                        'page': page_num,
                        'chunk_num': i + 1,
                        'source': file_path
                    }
                })
        
        return chunks
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF artifacts
        text = re.sub(r'([a-z])-\s+([a-z])', r'\1\2', text)
        
        # Remove header/footer artifacts
        text = re.sub(r'\d+\s+of\s+\d+', '', text)
        
        return text.strip()
    
    def create_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Create overlapping chunks preserving structure."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Find end of chunk
            end = start + chunk_size
            
            if end < len(text):
                # Try to end at sentence boundary
                next_period = text.find('.', end - 50, end + 50)
                if next_period != -1:
                    end = next_period + 1
            
            # Get chunk
            chunk = text[start:end].strip()
            
            # Only add if meets minimum size
            if len(chunk) >= self.config['min_chunk_size']:
                chunks.append(chunk)
            
            # Move start position
            start = end - overlap
        
        return chunks
```

### 2. Layout-Aware Processing

```python
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

class LayoutAwarePDFProcessor(PDFProcessor):
    def process_pdf(self, file_path: str) -> List[Dict]:
        """Process PDF preserving layout elements."""
        chunks = []
        
        # Extract pages with layout
        for page_num, page_layout in enumerate(
            extract_pages(file_path), 1
        ):
            # Extract text elements
            elements = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    elements.append({
                        'text': element.get_text(),
                        'bbox': element.bbox,
                        'type': type(element).__name__
                    })
            
            # Sort by position (top to bottom, left to right)
            elements.sort(
                key=lambda e: (-e['bbox'][1], e['bbox'][0])
            )
            
            # Combine elements into coherent text
            page_text = self.combine_elements(elements)
            
            # Create chunks
            page_chunks = self.create_chunks(
                page_text,
                self.config['chunk_size'],
                self.config['chunk_overlap']
            )
            
            # Add metadata
            for i, chunk in enumerate(page_chunks):
                chunks.append({
                    'text': chunk,
                    'metadata': {
                        'page': page_num,
                        'chunk_num': i + 1,
                        'source': file_path,
                        'layout_preserved': True
                    }
                })
        
        return chunks
    
    def combine_elements(self, elements: List[Dict]) -> str:
        """Combine text elements respecting layout."""
        text = ""
        current_line_y = None
        
        for elem in elements:
            # Check if new line needed
            if current_line_y is not None:
                if abs(elem['bbox'][1] - current_line_y) > 5:
                    text += "\n"
            
            # Add text
            text += elem['text'].strip() + " "
            current_line_y = elem['bbox'][1]
        
        return text.strip()
```

## Usage Example

### 1. Basic Processing

```python
# Initialize processor
processor = PDFProcessor({
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'clean_text': True
})

# Process PDF
file_path = "example.pdf"
chunks = processor.process_pdf(file_path)

print(f"Processed {len(chunks)} chunks")

# Preview chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1}:")
    print(f"Page: {chunk['metadata']['page']}")
    print(f"Text: {chunk['text'][:100]}...")
```

### 2. Layout-Aware Processing

```python
# Initialize layout-aware processor
processor = LayoutAwarePDFProcessor({
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'preserve_layout': True
})

# Process PDF
chunks = processor.process_pdf("example.pdf")

# Print structure
for chunk in chunks[:2]:
    print(f"\nPage {chunk['metadata']['page']}:")
    print(chunk['text'][:100])
```

## Advanced Features

### 1. Table Detection

```python
import tabula

def extract_tables(pdf_path):
    """Extract tables from PDF."""
    # Read tables
    tables = tabula.read_pdf(
        pdf_path,
        pages='all',
        multiple_tables=True
    )
    
    # Convert to text
    table_texts = []
    for i, table in enumerate(tables, 1):
        text = f"Table {i}:\n"
        text += table.to_string()
        table_texts.append(text)
    
    return table_texts
```

### 2. Image Handling

```python
from pdf2image import convert_from_path
import pytesseract

def extract_images(pdf_path):
    """Extract and OCR images from PDF."""
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    
    # Process each image
    image_texts = []
    for i, image in enumerate(images, 1):
        # OCR the image
        text = pytesseract.image_to_string(image)
        
        if text.strip():
            image_texts.append({
                'page': i,
                'text': text.strip()
            })
    
    return image_texts
```

## Best Practices

1. **Text Extraction**
   - Handle different PDF formats
   - Preserve important whitespace
   - Remove artifacts carefully

2. **Chunking Strategy**
   - Respect semantic boundaries
   - Maintain context windows
   - Handle short sections

3. **Metadata**
   - Track page numbers
   - Preserve document structure
   - Include source information

## Common Issues

1. **Poor Text Extraction**
   - Try different PDF libraries
   - Check PDF encoding
   - Use OCR for scanned docs

2. **Layout Problems**
   - Adjust parsing parameters
   - Consider column detection
   - Handle headers/footers

3. **Special Content**
   - Extract tables separately
   - Process images with OCR
   - Handle equations carefully

## Next Steps

- Explore [Custom Knowledge Base](custom-kb.md)
- Learn about [Evaluation](evaluation.md)
- See [Deployment Options](../deployment/local.md)