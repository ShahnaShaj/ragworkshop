# Generator / LLM Integration

This section explains how we integrate with Google's Gemini model for generating answers based on retrieved context.

## Overview

The generator component:
1. Takes retrieved context and user query
2. Constructs an appropriate prompt
3. Calls the Gemini API
4. Processes and returns the response

## Implementation

### Basic Generator

```python
import google.generativeai as genai

class Generator:
    def __init__(self, model_name='gemini-2.5-flash'):
        self.model = genai.GenerativeModel(model_name)
        
    def generate(self, query, contexts, temperature=0.3):
        """
        Generate an answer using retrieved contexts.
        
        Args:
            query (str): User question
            contexts (list): Retrieved text passages
            temperature (float): Generation temperature
            
        Returns:
            str: Generated answer
        """
        # Build prompt
        prompt = self._build_prompt(query, contexts)
        
        # Generate response
        response = self.model.generate_content(
            prompt,
            temperature=temperature
        )
        
        return response.text
    
    def _build_prompt(self, query, contexts):
        """Build a prompt from query and contexts."""
        context_text = "\n\n".join(
            f"[{i+1}] {ctx}" 
            for i, ctx in enumerate(contexts)
        )
        
        return f"""Based on the following passages, answer the question.
        If you cannot find relevant information, say so.
        Include passage numbers [1], [2], etc. to cite your sources.

        Passages:
        {context_text}

        Question: {query}

        Answer:"""
```

### Advanced Generator

```python
class AdvancedGenerator(Generator):
    def __init__(self, model_name='gemini-2.5-flash'):
        super().__init__(model_name)
        self.history = []
        
    def generate(self, query, contexts, 
                temperature=0.3,
                max_tokens=1000,
                citation_required=True):
        """
        Enhanced generation with history and citations.
        """
        # Build conversation history
        messages = self._build_conversation(
            query, contexts, citation_required
        )
        
        # Generate with safety settings
        response = self.model.generate_content(
            messages,
            generation_config={
                'temperature': temperature,
                'max_output_tokens': max_tokens,
                'top_p': 0.9,
                'top_k': 40
            },
            safety_settings={
                'harmful_categories': 'block'
            }
        )
        
        # Update history
        self.history.append({
            'query': query,
            'contexts': contexts,
            'response': response.text
        })
        
        return self._format_response(response.text)
    
    def _build_conversation(self, query, contexts, 
                          citation_required):
        """Build conversation with history."""
        system_prompt = """You are a helpful AI assistant.
        Answer questions based only on the provided context.
        Be concise and accurate."""
        
        if citation_required:
            system_prompt += "\nCite passages using [1], [2] etc."
            
        context_text = self._format_contexts(contexts)
        
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Context:\n{context_text}"},
            {'role': 'user', 'content': f"Question: {query}"}
        ]
```

## Prompt Engineering

### Basic Prompt Template

```python
BASIC_PROMPT_TEMPLATE = """Answer the question based on these passages:

{contexts}

Question: {query}

Answer:"""
```

### Enhanced Prompt Template

```python
ENHANCED_PROMPT_TEMPLATE = """You are a helpful AI assistant answering questions based on provided context.

CONTEXT:
{contexts}

QUESTION: {query}

Instructions:
- Use ONLY information from the provided context
- Cite sources using [1], [2], etc.
- If context doesn't contain relevant info, say "I cannot answer based on the provided context"
- Be concise and direct
- Format lists and technical terms appropriately

ANSWER:"""
```

## Configuration

Generator settings in `config.yaml`:

```yaml
generator:
  model: gemini-2.5-flash
  temperature: 0.3
  max_tokens: 1000
  citation_required: true
  prompt:
    template: enhanced
    system_message: true
  safety:
    harmful_content: block
    unsafe_content: harmless
```

## Response Processing

### Citation Extraction

```python
import re

def extract_citations(text):
    """Extract passage citations from response."""
    citations = re.findall(r'\[(\d+)\]', text)
    return [int(c) for c in citations]

def validate_citations(text, num_contexts):
    """Validate citation numbers."""
    citations = extract_citations(text)
    return all(1 <= c <= num_contexts for c in citations)
```

### Response Formatting

```python
def format_response(text, contexts):
    """Format response with citations."""
    citations = extract_citations(text)
    used_contexts = [contexts[i-1] for i in citations]
    
    return {
        'answer': text,
        'citations': citations,
        'contexts': used_contexts
    }
```

## Error Handling

```python
class GeneratorError(Exception):
    """Base class for generator errors."""
    pass

class ContextError(GeneratorError):
    """Error when context is invalid."""
    pass

class APIError(GeneratorError):
    """Error when API call fails."""
    pass

def handle_generation_errors(func):
    """Decorator for error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except genai.APIError as e:
            raise APIError(f"API error: {str(e)}")
        except Exception as e:
            raise GeneratorError(f"Generation failed: {str(e)}")
    return wrapper
```

## Best Practices

### Prompt Design
- Clear instructions
- Consistent format
- Citation requirements
- Error handling guidance

### API Usage
- Proper error handling
- Rate limiting
- Retry logic
- Response validation

### Output Quality
- Citation validation
- Response formatting
- Length control
- Safety checks

## Example Usage

```python
# Initialize generator
generator = AdvancedGenerator()

# Generate answer
query = "What is RAG?"
contexts = [
    "RAG (Retrieval-Augmented Generation) is a technique...",
    "This approach combines retrieval with generation..."
]

response = generator.generate(
    query=query,
    contexts=contexts,
    temperature=0.3,
    citation_required=True
)

print(response)
```

## Troubleshooting

### API Errors
- Check API key
- Verify rate limits
- Validate input format

### Quality Issues
- Adjust temperature
- Refine prompt
- Check context quality

### Performance
- Cache responses
- Batch requests
- Monitor latency

## Next Steps

### Further Reading
- End-to-End Pipeline
- Example Queries
- Evaluation Metrics