# Getting Started

This guide will help you set up and run the RAG Workshop project quickly.

## Prerequisites

### Python Environment
- Python 3.7 or later
- Virtual environment (conda or venv) recommended

### Required Libraries
- `google-generativeai`: Google's Gemini API client
- `pypdf`: PDF processing
- `sentence-transformers`: Document embedding generation
- `faiss-cpu`: Vector similarity search
- `gradio`: Web interface creation

### API Keys
1. **Google Gemini API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Save it securely

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-workshop.git
cd rag-workshop
```

### 2. Create a Virtual Environment
=== "Using venv"
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    source venv/bin/activate  # Linux/Mac
    ```

=== "Using conda"
    ```bash
    conda create -n rag-workshop python=3.9
    conda activate rag-workshop
    ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in your project root:
```bash
GEMINI_API_KEY=your_api_key_here
```

## Quick Start Example

### 1. Launch Jupyter Notebook
```bash
jupyter notebook rag_workshop.ipynb
```

### 2. Run the Example
Execute the following code in your notebook:

```python
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize model
model = genai.GenerativeModel('gemini-2.5-flash')

# Test the setup
response = model.generate_content("Hello! Can you verify that you're working?")
print(response.text)
```

If you see a response from the model, congratulations! Your setup is complete.

## Next Steps
- Learn about the [Project Overview](overview.md)
- Explore the [Code Walkthrough](code/data-loading.md)
- Try out the [Examples](examples/basic-retrieval.md)